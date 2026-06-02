import os
import json
import torch
import argparse
import numpy as np
from datetime import datetime
from engine.logger import Logger
from engine.solver import Trainer
from data.build_dataloader import build_dataloader, build_dataloader_conditional
from utils.conditional_evaluation import compute_conditional_metrics, save_conditional_visualizations
from utils.evaluation_common import append_metrics_csv, update_total_metrics_csv
from utils.runtime import (
    instantiate_from_config,
    load_yaml_config,
    merge_opts_to_config,
    seed_everything,
    unnormalize_to_zero_to_one,
)
from utils.unconditional_evaluation import compute_unconditional_metrics, save_unconditional_visualizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None,help='experiment name')
    parser.add_argument('--checkpoint_name', type=str, default=None,
                        help='Experiment name used to load checkpoints during evaluation. Defaults to --name.')
    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='output',
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
                        help='Generation mode: uncond or conditional.')
    parser.add_argument('--milestone', type=int, default=5,help='Checkpoint num')

    # args for config overrides
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--model_name', type=str, default='pacodi',
                        help='The name of the model, used for path naming')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='Dataset name used for path grouping, e.g. etth in pacodi_sde/etth/etth_seq128.')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='How many times to run the sampling process')
    parser.add_argument('--visualization_compare', type=int, default=3000,
                        help='Number of real/synthetic samples used in PCA/t-SNE/KDE visualizations.')
    parser.add_argument('--eval_sample_limit', type=int, default=None,
                        help='Maximum number of generated samples during evaluation. Defaults to the full dataset.')
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip metric computation during checkpoint evaluation.')
    parser.add_argument('--metric_samples', type=int, default=3000,
                        help='Maximum number of paired real/synthetic samples used for metrics.')
    parser.add_argument('--discriminative_iterations', type=int, default=2000,
                        help='Training steps for the post-hoc discriminative metric.')
    parser.add_argument('--predictive_iterations', type=int, default=5000,
                        help='Training steps for the post-hoc predictive metric.')

    args = parser.parse_args()

    if args.name is None:
        raise ValueError("--name is required for experiment path naming.")

    if args.dataset_name is None:
        args.dataset_name = args.name.split("_seq", 1)[0].split("_", 1)[0]

    args.checkpoint_name = args.name if args.train or args.checkpoint_name is None else args.checkpoint_name
    args.experiment_dir = os.path.join('experiments', args.mode, args.model_name, args.dataset_name, args.name)
    args.checkpoint_experiment_dir = os.path.join(
        'experiments',
        args.mode,
        args.model_name,
        args.dataset_name,
        args.checkpoint_name,
    )
    args.checkpoint_folder = os.path.join(args.checkpoint_experiment_dir, 'checkpoint')
    args.save_dir = os.path.join(args.experiment_dir, 'log')
    args.savesample_dir = os.path.join(args.experiment_dir, 'samples')
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

    def load_unconditional_reference(dataset):
        truth_dir = os.path.join(args.savesample_dir, 'truth')
        norm_path = os.path.join(truth_dir, 'norm_truth_train.npy')
        raw_path = os.path.join(truth_dir, 'ground_truth_train.npy')
        if os.path.exists(norm_path):
            return np.load(norm_path)
        if os.path.exists(raw_path):
            return np.load(raw_path)
        return dataset.samples

    def save_visualizations(real_data, fake_data, output_dir):
        paths = save_unconditional_visualizations(
            real_data,
            fake_data,
            output_dir,
            compare=args.visualization_compare,
            seed=args.seed,
        )
        if logger is not None:
            logger.log_info(f"Saved distribution visualizations: {paths}")

    def save_conditional_visuals(real_data, fake_data, output_dir):
        paths = save_conditional_visualizations(
            real_data,
            fake_data,
            output_dir,
            compare=args.visualization_compare,
            seed=args.seed,
        )
        if logger is not None:
            logger.log_info(f"Saved conditional visualizations: {paths}")

    def metric_base_row():
        training_timing = {}
        timing_path = os.path.join(args.checkpoint_experiment_dir, 'training_timing.json')
        if os.path.exists(timing_path):
            with open(timing_path, 'r', encoding='utf-8') as f:
                training_timing = json.load(f)

        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': args.mode,
            'model_name': args.model_name,
            'dataset_name': args.dataset_name,
            'experiment': args.name,
            'checkpoint_experiment': args.checkpoint_name,
            'milestone': args.milestone,
            'seq_length': config['model']['params'].get('seq_length'),
            'channels': config['model']['params'].get('channels'),
            'backbone': config['model']['params'].get('backbone'),
            'real_imag_interaction': config['model']['params'].get('real_imag_interaction'),
            'instance_norm': config['model']['params'].get('instance_norm'),
            'frequency_keep_ratio': config['model']['params'].get('frequency_keep_ratio', 1.0),
            'frequency_keep_bins': config['model']['params'].get('frequency_keep_bins'),
            'ddim_eta': config['model']['params'].get('ddim_eta'),
            'sampling_steps': getattr(trainer.ema.ema_model, 'sampling_steps', config['model']['params'].get('sampling_steps')),
            'train_total_steps': training_timing.get('train_total_steps'),
            'train_timing_steps': training_timing.get('train_timing_steps'),
            'train_100_steps_time_sec': training_timing.get('train_100_steps_time_sec'),
            'train_time_per_step_sec': training_timing.get('train_time_per_step_sec'),
            'train_total_time_sec': training_timing.get('train_total_time_sec'),
            'inference_time_sec': trainer.last_sample_time_seconds,
            'inference_samples': trainer.last_sample_count,
            'inference_samples_per_sec': trainer.last_sample_throughput,
        }

    def save_metrics(metrics):
        row = metric_base_row()
        row.update(metrics)
        metrics_path = os.path.join(args.experiment_dir, 'metrics.csv')
        total_path = os.path.join('experiments', args.mode, args.model_name, args.dataset_name, 'total.csv')
        saved_path = append_metrics_csv(metrics_path, row)
        saved_total_path = update_total_metrics_csv(total_path, row)
        if logger is not None:
            logger.log_info(f"Saved metrics to {saved_path}: {metrics}")
            logger.log_info(f"Updated dataset total metrics at {saved_total_path}")

    # Data preparation
    if args.mode == 'uncond':
        dataloader_info = build_dataloader(config, args)
        test_dataloader_info = None
    elif args.mode == 'conditional':
        dataloader_info = build_dataloader_conditional(config, args, period='train')
        test_dataloader_info = build_dataloader_conditional(config, args, period='test')
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
            eval_sampling_steps = config['dataloader'].get('test_dataset', {}).get(
                'sampling_steps',
                config['model']['params'].get('sampling_steps', None),
            )
            if eval_sampling_steps is not None:
                trainer.model.sampling_steps = int(eval_sampling_steps)
                trainer.ema.ema_model.sampling_steps = int(eval_sampling_steps)
                logger.log_info(f"Use evaluation sampling_steps={int(eval_sampling_steps)}")

            dataset = dataloader_info['dataset']
            eval_sample_count = len(dataset)
            if args.eval_sample_limit is not None:
                eval_sample_count = min(eval_sample_count, int(args.eval_sample_limit))
                logger.log_info(f"Use eval_sample_limit={eval_sample_count}")
            samples = trainer.sample(
                num=eval_sample_count,
                size_every=config['dataloader'].get('sample_size', 256),
                shape=[dataset.window, dataset.var_num],
            )
            samples = maybe_unnormalize(samples, dataset)
            save_path = os.path.join(args.savesample_dir, 'samples.npy')
            np.save(save_path, samples)
            real_data = load_unconditional_reference(dataset)
            save_visualizations(real_data, samples, args.savesample_dir)
            if not args.skip_metrics:
                metrics = compute_unconditional_metrics(
                    real_data,
                    samples,
                    max_samples=args.metric_samples,
                    discriminative_iterations=args.discriminative_iterations,
                    predictive_iterations=args.predictive_iterations,
                    seed=args.seed,
                    device=f"cuda:{args.gpu}" if args.gpu is not None else None,
                )
                save_metrics(metrics)

        elif args.mode == 'conditional':
            test_dataloader = test_dataloader_info['dataloader']
            test_dataset = test_dataloader_info['dataset']
            sampling_steps = config['model']['params'].get('sampling_steps', 50)

            for run_id in range(args.num_samples):
                print(f">>> Running sampling loop {run_id + 1}/{args.num_samples}...")
                samples, reals = trainer.conditional_sample(
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

                if run_id == 0:
                    save_conditional_visuals(reals, samples, run_save_path)
                    if not args.skip_metrics:
                        metrics = compute_conditional_metrics(reals, samples)
                        metrics['run_id'] = run_id
                        save_metrics(metrics)

if __name__ == '__main__':
    main()
