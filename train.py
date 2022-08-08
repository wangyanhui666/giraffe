from scipy import misc
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import argparse
import time
from pathlib import Path
from im2scene import config
from im2scene.checkpoints import CheckpointIO
import logging
import datetime

import util.misc as misc
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP


def get_args_parser():
    parser = argparse.ArgumentParser('video giraffe training', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='generator Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--batch_size_d',default=32,type=int,
                        help='discriminator Batch size per GPU')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate of generator (absolute lr)')
    parser.add_argument('--lr_d', type=float, default=None, metavar='LR',
                        help='learning rate of discriminator (absolute lr)')
    parser.add_argument('--blr', type=float, default=5e-4, metavar='LR',
                        help='base learning rate of generator: absolute_lr = base_lr * total_batch_size / 32')
    parser.add_argument('--blr_d', type=float, default=1e-4, metavar='LR',
                        help='base learning rate of discriminator: absolute_lr = base_lr * total_batch_size / 32')                    
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # config file
    parser.add_argument('--config',type=str, help='Path to config file.')
    parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
    parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of '
                         'seconds with exit code 2.')

    return parser



def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device=torch.device(args.device)

    # fix the seed for reproducibility
    seed=args.seed+misc.get_rank()
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True

    logger_py = logging.getLogger(__name__)

    cfg = config.load_config(args.config, 'configs/default.yaml')
    # Shorthands
    # args.output_dir = cfg['training']['args.output_dir']
    backup_every = cfg['training']['backup_every']
    exit_after = args.exit_after
    args.blr = cfg['training']['learning_rate']
    args.blr_d = cfg['training']['learning_rate_d']
    t0 = time.time()

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    eff_batch_size_d = args.batch_size_d * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 32
    if args.lr_d is None:  # only base_lr is specified
        args.lr_d = args.blr_d * eff_batch_size_d / 32
    print("base lr: %.2e" % (args.lr * 32 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("base lr_d: %.2e" % (args.lr_d * 32 / eff_batch_size_d))
    print("actual lr_d: %.2e" % args.lr_d)
    model_selection_metric = cfg['training']['model_selection_metric']
    if cfg['training']['model_selection_mode'] == 'maximize':
        model_selection_sign = 1
    elif cfg['training']['model_selection_mode'] == 'minimize':
        model_selection_sign = -1
    else:
        raise ValueError('model_selection_mode must be '
                        'either maximize or minimize.')

    # Output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # dataset
    dataset_train = config.get_dataset(cfg)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # logdir
    args.log_dir=os.path.join(args.output_dir,'log')
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size_d,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # model
    model = config.get_model(cfg, device=device, len_dataset=len(dataset_train),args=args)
    model.to(device)
    model_without_ddp=model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    # Print model
    nparameters = sum(p.numel() for p in model.parameters())
    logger_py.info(model)
    logger_py.info('Total number of parameters: %d' % nparameters)

    if hasattr(model, "discriminator") and model.discriminator is not None:
        nparameters_d = sum(p.numel() for p in model.discriminator.parameters())
        logger_py.info(
            'Total number of discriminator parameters: %d' % nparameters_d)
    if hasattr(model, "generator") and model.generator is not None:
        nparameters_g = sum(p.numel() for p in model.generator.parameters())
        logger_py.info('Total number of generator parameters: %d' % nparameters_g)


    # Optimizer
    op = optim.RMSprop if cfg['training']['optimizer'] == 'RMSprop' else optim.Adam
    optimizer_kwargs = cfg['training']['optimizer_kwargs']

    if hasattr(model, "generator") and model.generator is not None:
        parameters_g = model.generator.parameters()
    else:
        parameters_g = list(model.decoder.parameters())
    optimizer = op(parameters_g, lr=args.lr, **optimizer_kwargs)

    if hasattr(model, "discriminator") and model.discriminator is not None:
        parameters_d = model.discriminator.parameters()
        optimizer_d = op(parameters_d, lr=args.lr_d)
    else:
        optimizer_d = None
    

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # bug to fix-----------------------------------
    checkpoint_io = CheckpointIO(args.output_dir, model=model_without_ddp, optimizer=optimizer,
                             optimizer_d=optimizer_d)
    if args.resume:
        checkpoint=torch.load(args.checkpoint, map_location='cpu')
        msg=model.load_state_dict(checkpoint)
        print(msg)
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,loss_scaler=loss_scaler)
    try:
        load_dict = checkpoint_io.load('model.pt')
        print("Loaded model checkpoint.")
    except FileExistsError:
        load_dict = dict()
        print("No model checkpoint found.")

    # trainer
    trainer = config.get_trainer(model.module, optimizer, optimizer_d, cfg, device=device,args=args)

    # prepare training------------------------------
    epoch_it = load_dict.get('epoch_it', -1)
    it = load_dict.get('it', -1)
    metric_val_best = load_dict.get(
        'loss_val_best', -model_selection_sign * np.inf)

    if metric_val_best == np.inf or metric_val_best == -np.inf:
        metric_val_best = -model_selection_sign * np.inf

    print('Current best validation metric (%s): %.8f'
        % (model_selection_metric, metric_val_best))


    # Shorthands
    print_every = cfg['training']['print_every']
    checkpoint_every = cfg['training']['checkpoint_every']
    validate_every = cfg['training']['validate_every']
    visualize_every = cfg['training']['visualize_every']

    start_time=time.time()
    t0b = time.time()

    while (True):
        epoch_it += 1
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch_it)
        for batch in data_loader_train:

            it += 1
            loss = trainer.train_step(batch, it)
            if log_writer is not None:
                for (k, v) in loss.items():
                    log_writer.add_scalar(k, v, it)
            # Print output
            if misc.is_main_process():
                if print_every > 0 and (it % print_every) == 0:
                    info_txt = '[Epoch %02d] it=%03d, time=%.3f' % (
                        epoch_it, it, time.time() - t0b)
                    for (k, v) in loss.items():
                        info_txt += ', %s: %.4f' % (k, v)
                    logger_py.info(info_txt)
                    t0b = time.time()

                # # Visualize output
                if visualize_every > 0 and (it % visualize_every) == 0:
                    logger_py.info('Visualizing')
                    image_grid = trainer.visualize(it=it)
                    if image_grid is not None:
                        log_writer.add_image('images', image_grid, it)

                # Save checkpoint
                if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
                    logger_py.info('Saving checkpoint')
                    print('Saving checkpoint')
                    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)

                # Backup if necessary
                if (backup_every > 0 and (it % backup_every) == 0):
                    logger_py.info('Backup checkpoint')
                    checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)

                # Run validation
                if validate_every > 0 and (it % validate_every) == 0 and (it > 0):
                    print("Performing evaluation step.")
                    eval_dict = trainer.evaluate()
                    metric_val = eval_dict[model_selection_metric]
                    logger_py.info('Validation metric (%s): %.4f'
                                % (model_selection_metric, metric_val))

                    for k, v in eval_dict.items():
                        log_writer.add_scalar('val/%s' % k, v, it)

                    if model_selection_sign * (metric_val - metric_val_best) > 0:
                        metric_val_best = metric_val
                        logger_py.info('New best model (loss %.4f)' % metric_val_best)
                        checkpoint_io.backup_model_best('model_best.pt')
                        checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                        loss_val_best=metric_val_best)

                # Exit if necessary
                if exit_after > 0 and (time.time() - t0) >= exit_after:
                    logger_py.info('Time limit reached. Exiting.')
                    checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                                    loss_val_best=metric_val_best)
                    exit(3)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)