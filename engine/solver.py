import os
import sys
import time
import copy
import json
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from utils.ema import EMA
from utils.runtime import get_model_parameters_info, instantiate_from_config



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
        self.train_num_steps = config['solver']['train_steps']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.checkpoint_count = int(config['solver'].get('checkpoint_count', 5))
        self.save_cycle = int(config['solver'].get(
            'save_cycle',
            max(1, self.train_num_steps // self.checkpoint_count),
        ))
        self.dl = cycle(dataloader['dataloader'])
        self.dataloader = dataloader['dataloader']
        self.dataset = dataloader.get('dataset')
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
        self.train_timing_steps = int(config['solver'].get('timing_steps', 100))
        self.train_timing_path = Path(args.checkpoint_experiment_dir) / "training_timing.json"
        self.last_sample_time_seconds = None
        self.last_sample_count = None
        self.last_sample_throughput = None

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _save_training_timing(self, timing):
        self.train_timing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.train_timing_path, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)

    def _instance_stats_from_reference(self, start, batch_size):
        ema_model = self.ema.ema_model
        if not getattr(ema_model, "instance_norm", False) or self.dataset is None:
            return None

        reference = self.dataset.samples[start:start + batch_size]
        if len(reference) < batch_size:
            repeat = batch_size - len(reference)
            reference = np.concatenate([reference, self.dataset.samples[:repeat]], axis=0)

        reference = torch.from_numpy(reference).to(self.device).float()
        _, instance_stats = ema_model.instance_normalize(reference)
        return instance_stats

    def _conditional_instance_stats(self, x_raw, condition):
        ema_model = self.ema.ema_model
        if not getattr(ema_model, "instance_norm", False):
            return None

        if x_raw is None:
            return None

        _, instance_stats = ema_model.instance_normalize(x_raw)
        return instance_stats

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
        timing_start_step = self.step
        timing_target_step = min(self.train_num_steps, timing_start_step + self.train_timing_steps)
        timing_start_time = None
        train_100_steps_time = None
        data_iter = iter(self.dl)
        self.model.train()

        with tqdm(initial=self.step, total=self.train_num_steps, desc='Training') as pbar:
            while self.step < self.train_num_steps:
                if timing_start_time is None and self.step < timing_target_step:
                    self._sync_device()
                    timing_start_time = time.time()

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

                grad_norm = clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss_step)
                self.opt.zero_grad()
                self.ema.update()
                self.step += 1

                if train_100_steps_time is None and self.step >= timing_target_step and timing_start_time is not None:
                    self._sync_device()
                    train_100_steps_time = time.time() - timing_start_time
                    measured_steps = self.step - timing_start_step
                    if self.logger is not None:
                        self.logger.log_info(
                            f"Training timing. Steps: {measured_steps} | Time: {train_100_steps_time:.2f}s"
                        )
                        self.logger.add_scalar(
                            tag="train/time_per_100_steps",
                            scalar_value=train_100_steps_time,
                            global_step=self.step,
                        )

                if self.logger is not None:
                    self.logger.add_scalar(tag="train/loss", scalar_value=total_loss_step, global_step=self.step)
                    self.logger.add_scalar(
                        tag="train/lr",
                        scalar_value=self.opt.param_groups[0]["lr"],
                        global_step=self.step,
                    )
                    self.logger.add_scalar(
                        tag="train/grad_norm",
                        scalar_value=float(grad_norm.detach().cpu()),
                        global_step=self.step,
                    )

                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)

                pbar.update(1)

        total_duration = time.time() - total_training_start
        if train_100_steps_time is None and timing_start_time is not None:
            self._sync_device()
            train_100_steps_time = time.time() - timing_start_time

        measured_steps = max(0, min(self.step, timing_target_step) - timing_start_step)
        timing = {
            "train_total_steps": int(self.step),
            "train_total_time_sec": float(total_duration),
            "train_timing_steps": int(measured_steps),
            "train_100_steps_time_sec": float(train_100_steps_time) if train_100_steps_time is not None else None,
            "train_time_per_step_sec": (
                float(train_100_steps_time) / measured_steps
                if train_100_steps_time is not None and measured_steps > 0
                else None
            ),
        }
        self._save_training_timing(timing)
        print('training complete')
        if self.logger is not None:
            self.logger.log_info(f'Training Done. Total Steps: {self.step} | Time: {total_duration:.2f}s')
            self.logger.log_info(f"Saved training timing to {self.train_timing_path}")

    def sample(self, num, size_every, shape=None, model_kwargs=None, cond_fn=None):
        self._sync_device()
        tic = time.time()
        if self.logger is not None:
            self.logger.log_info('Begin to sample...')

        all_samples = []
        current_num = 0

        while current_num < num:
            batch_size = min(size_every, num - current_num)
            instance_stats = self._instance_stats_from_reference(current_num, batch_size)

            sample = self.ema.ema_model.generate_unconditional(
                batch_size=batch_size,
                instance_stats=instance_stats,
            )

            all_samples.append(sample.detach().cpu().numpy())
            current_num += batch_size
            torch.cuda.empty_cache()

        samples = np.concatenate(all_samples, axis=0)
        self._sync_device()
        self.last_sample_time_seconds = time.time() - tic
        self.last_sample_count = int(samples.shape[0])
        self.last_sample_throughput = (
            self.last_sample_count / self.last_sample_time_seconds
            if self.last_sample_time_seconds > 0
            else None
        )

        if self.logger is not None:
            self.logger.log_info(
                'Sampling done, time: {:.2f}s, throughput: {:.2f} samples/s'.format(
                    self.last_sample_time_seconds,
                    self.last_sample_throughput or 0.0,
                )
            )
        return samples

    def conditional_sample(self, num, size_every, raw_dataloader, shape=None, sampling_steps=50):
        self._sync_device()
        tic = time.time()
        all_samples = []
        all_reals = []
        self.ema.ema_model.eval()

        current_count = 0
        device = self.device

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(raw_dataloader, desc="Conditional Sampling")):
                if current_count >= num:
                    break

                if isinstance(batch, (list, tuple)):
                    x_raw = batch[0].to(device).float()
                    cond = batch[1].to(device).float()
                else:
                    cond = batch.to(device).float()
                    x_raw = None

                actual_batch_size = cond.shape[0]
                instance_stats = self._conditional_instance_stats(x_raw, cond)

                sample = self.ema.ema_model.generate_conditional(
                    batch_size=actual_batch_size,
                    cond=cond,
                    sampling_steps=sampling_steps,
                    instance_stats=instance_stats,
                )

                all_samples.append(sample.detach().cpu().numpy())
                if x_raw is not None:
                    all_reals.append(x_raw.detach().cpu().numpy())

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

        self._sync_device()
        self.last_sample_time_seconds = time.time() - tic
        self.last_sample_count = int(samples.shape[0])
        self.last_sample_throughput = (
            self.last_sample_count / self.last_sample_time_seconds
            if self.last_sample_time_seconds > 0
            else None
        )
        if self.logger is not None:
            self.logger.log_info(
                'Conditional sampling done, time: {:.2f}s, throughput: {:.2f} samples/s'.format(
                    self.last_sample_time_seconds,
                    self.last_sample_throughput or 0.0,
                )
            )

        return samples, reals


