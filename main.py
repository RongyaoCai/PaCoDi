import os
import torch
import argparse
import numpy as np
from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_text
from Utils.io_utils import load_yaml_config,seed_everything,merge_opts_to_config,instantiate_from_config,unnormalize_to_zero_to_one
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None,help='experiment name')
    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for randomness
    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='uncond',
                        help='Uncond, Infilling or Forecasting or Text.')
    parser.add_argument('--milestone', type=int, default=10,help='Checkpoint num')

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for config overrides
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--model_name', type=str, default='pacodi',
                        help='The name of the model, used for path naming')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='How many times to run the sampling process')

    args = parser.parse_args()

    args.checkpoint_folder = os.path.join('experiments', args.mode, args.model_name, args.name,'checkpoint')
    args.save_dir = os.path.join('experiments',args.mode,args.model_name,args.name, 'log')
    args.savesample_dir = os.path.join('experiments', args.mode, args.model_name, args.name, 'samples')
    os.makedirs(args.checkpoint_folder, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.savesample_dir, exist_ok=True)

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()

    def maybe_unnormalize(samples, dataset):
        if dataset.auto_norm:
            return unnormalize_to_zero_to_one(samples)
        return samples

    # Data preparation
    if args.mode == 'uncond':
        dataloader_info = build_dataloader(config, args)
        test_dataloader_info = None
    elif args.mode == 'text':
        dataloader_info = build_dataloader_text(config, args, period='train')
        test_dataloader_info = build_dataloader_text(config, args, period='test')
    else:
        raise ValueError(f"Unrecognized mode: {args.mode}. Please check the input arguments.")

    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    # Train
    if args.train:
        trainer.train()

    # Test generation
    else:
        trainer.load(args.milestone)
        if args.mode == 'uncond':
            dataset = dataloader_info['dataset']
            samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
            samples = maybe_unnormalize(samples, dataset)
            save_path = os.path.join(args.savesample_dir, 'samples.npy')
            np.save(save_path, samples)

        elif args.mode == 'text':
            test_dataloader = test_dataloader_info['dataloader']
            test_dataset = test_dataloader_info['dataset']
            sampling_steps = config['model']['params'].get('sampling_timesteps', 50)

            for run_id in range(args.num_samples):
                print(f">>> Running sampling loop {run_id + 1}/{args.num_samples}...")
                samples, reals = trainer.text_sample(
                    num=len(test_dataset),
                    size_every=config['dataloader'].get('sample_size', 16),
                    raw_dataloader=test_dataloader,
                    sampling_steps=sampling_steps
                )

                if run_id == 0:
                    reals_physical = test_dataset.unnormalize(reals)
                    np.save(os.path.join(args.savesample_dir, 'reals_physical.npy'), reals_physical)
                    np.save(os.path.join(args.savesample_dir, 'reals.npy'), reals)

                samples_physical = test_dataset.unnormalize(samples)
                run_save_path = os.path.join(args.savesample_dir, f'run_{run_id}')
                os.makedirs(run_save_path, exist_ok=True)
                np.save(os.path.join(run_save_path, 'samples_physical.npy'), samples_physical)
                np.save(os.path.join(run_save_path, 'samples.npy'), samples)

if __name__ == '__main__':
    main()
