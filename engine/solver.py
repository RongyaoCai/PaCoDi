import os
import sys
import time
import copy
import torch
import numpy as np
import wandb
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from transformers.models.siglip.modeling_siglip import SiglipTextModelOutput
from Utils.io_utils import instantiate_from_config, get_model_parameters_info



sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        if args.gpu is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{args.gpu}")
        else:
            try:
                self.device = next(self.model.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger
        self.results_folder = Path(args.checkpoint_folder)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = copy.deepcopy(config['solver']['scheduler'])
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def save(self, milestone, verbose=False):
        checkpoint_path = self.results_folder / f'checkpoint-{milestone}.pt'
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(checkpoint_path)))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(checkpoint_path))

    def load(self, milestone, verbose=False):
        checkpoint_path = self.results_folder / f'checkpoint-{milestone}.pt'
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(checkpoint_path)))
        device = self.device
        data = torch.load(str(checkpoint_path), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        total_training_start = time.time()
        data_iter = iter(self.dl)
        self.model.train()

        with tqdm(initial=self.step, total=self.train_num_steps, desc='Training') as pbar:
            while self.step < self.train_num_steps:
                total_loss_step = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(data_iter)

                    if isinstance(data, (list, tuple)):
                        data = [item.to(device) for item in data]
                    else:
                        data = data.to(device)
                    loss = self.model(data, target=data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()

                    total_loss_step += loss.item()

                pbar.set_description(f'loss: {total_loss_step:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss_step)
                self.opt.zero_grad()
                self.ema.update()
                self.step += 1

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                pbar.update(1)

        total_duration = time.time() - total_training_start
        print('training complete')
        if self.logger is not None:
            self.logger.log_info(f'Training Done. Total Steps: {self.step} | Time: {total_duration:.2f}s')

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')

        all_samples = []
        current_num = 0

        while current_num < num:
            batch_size = min(size_every, num - current_num)

            sample = self.ema.ema_model.generate_mts(
                batch_size=batch_size,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn
            )

            all_samples.append(sample.detach().cpu().numpy())
            current_num += batch_size
            torch.cuda.empty_cache()

        samples = np.concatenate(all_samples, axis=0)

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}'.format(time.time() - tic))
        return samples

    def text_sample(self, num, size_every, raw_dataloader, shape=None, sampling_steps=50):
        all_samples = []
        all_reals = []
        self.ema.ema_model.eval()

        current_count = 0
        device = self.device

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(raw_dataloader, desc="Text-Guided Sampling")):
                if current_count >= num:
                    break

                if isinstance(batch, (list, tuple)):
                    x_raw = batch[0].to(device).float()
                    cond = batch[1].to(device).float()
                else:
                    cond = batch.to(device).float()
                    x_raw = None

                actual_batch_size = cond.shape[0]

                sample, real = self.ema.ema_model.generate_text(
                    batch_size=actual_batch_size,
                    x_raw = x_raw,
                    cond=cond,
                    sampling_steps=sampling_steps
                )

                all_samples.append(sample.detach().cpu().numpy())
                if x_raw is not None:
                    all_reals.append(real.detach().cpu().numpy())

                current_count += actual_batch_size
                torch.cuda.empty_cache()

        if all_samples:
            samples = np.concatenate(all_samples, axis=0)[:num]
        else:
            samples = np.empty((0,))

        if all_reals:
            reals = np.concatenate(all_reals, axis=0)[:num]
        else:
            reals = np.empty((0,))

        return samples, reals




