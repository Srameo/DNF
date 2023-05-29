import os
import time
import datetime
import yaml
import git

import torch
import torch.backends.cudnn as cudnn

from timm.utils import AverageMeter

from utils import load_checkpoint, load_pretrained, save_checkpoint, save_image_torch, get_grad_norm
from utils.config import parse_options, copy_cfg, ordered_dict_to_dict
from utils.scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.metrics import get_psnr_torch, get_ssim_torch
from utils.loss import build_loss
from utils.logger import create_logger

from models import build_model
from datasets import build_train_loader, build_valid_loader, build_test_loader
from forwards import build_forwards, build_profile

from torch.utils.tensorboard import SummaryWriter

def main(config):
    writer = SummaryWriter(os.path.join(config['output'], 'tensorboard'))
    train_dataloader = build_train_loader(config['data'])
    if not config['testset_as_validset']:
        valid_dataloader =  build_valid_loader(config['data'], 1)
    else:
        valid_dataloader = build_test_loader(config['data'], 2)

    logger.info(f"Creating model:{config['name']}/{config['model']['type']}")
    model = build_model(config['model'])
    model.cuda()
    logger.info(str(model))
    profile_forward = build_profile(config)
    profile_model(config, profile_forward, model, train_dataloader, logger)

    optimizer = build_optimizer(config['train'], model)
    lr_scheduler = build_scheduler(config['train'], optimizer, len(train_dataloader))
    loss_list = build_loss(config['loss'])
    logger.info(str(loss_list))
    
    logger.info('Building forwards:')
    logger.info(f'Train forward: {config["train"]["forward_type"]}')
    logger.info(f'Test forward: {config["test"]["forward_type"]}')
    train_forward, test_forward = build_forwards(config)

    max_psnr = 0.0
    max_ssim = 0.0
    total_epochs = config['train']['early_stop'] if config['train']['early_stop'] is not None else config['train']['epochs']

    if config.get('throughput_mode', False):
        throughput(config, train_forward, model, valid_dataloader, logger)
        return

    # set auto resume
    if config['train']['auto_resume']:
        auto_resume_path = os.path.join(config['output'], 'checkpoints', 'checkpoint.pth')
        if os.path.exists(auto_resume_path):
            config['train']['resume'] = auto_resume_path
            logger.info(f'Auto resume: setting resume path to {auto_resume_path}')
    
    if config['train'].get('resume'):
        max_psnr = load_checkpoint(config, model, optimizer, lr_scheduler, logger)
        validate(config, test_forward, model, loss_list, valid_dataloader, config['train'].get('start_epoch', 0), writer)
        if config.get('eval_mode', False):
            return

    if config['train'].get('pretrained') and (not config['train'].get('resume')):
        load_pretrained(config, model, logger)
        validate(config, test_forward, model, loss_list, valid_dataloader, config['train'].get('start_epoch', 0), writer)
        if config.get('eval_mode', False):
            return

    logger.info("Start training")
    start_time = time.time()
    start = time.time()
    lr_scheduler.step(config['train'].get('start_epoch', 0))

    for epoch in range(config['train'].get('start_epoch', 0)+1, total_epochs+1):
        train_one_epoch(config, train_forward, model, loss_list, train_dataloader, optimizer, None, epoch, lr_scheduler, writer)
        if epoch % config['valid_per_epoch'] == 0 or (total_epochs - epoch) < 50:
            psnr, ssim, loss = validate(config, test_forward, model, loss_list, valid_dataloader, epoch, writer)
            max_psnr = max(max_psnr, psnr)
            max_ssim = max(max_ssim, ssim)
            writer.add_scalar('eval/max_psnr', max_psnr, epoch)
            writer.add_scalar('eval/max_ssim', max_ssim, epoch)
        else:
            psnr = 0
        save_checkpoint(config, epoch, model, max_psnr, optimizer, lr_scheduler, logger, is_best=(max_psnr==psnr))
        logger.info(f'Train: [{epoch}/{config["train"]["epochs"]}] Max Valid PSNR: {max_psnr:.4f}, Max Valid SSIM: {max_ssim:.4f}')
        logger.info(f"Train: [{epoch}/{config['train']['epochs']}] Total Time {datetime.timedelta(seconds=int(time.time()-start))}")
        start = time.time()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

@torch.no_grad()
def profile_model(config, profile_forward, model, data_loader, logger):
    if profile_forward is not None:
        data_iter = iter(data_loader)
        data = next(data_iter)
        del data_iter
        profile_forward(config, model, data, logger)

    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Params: {n_parameters:,}")

def train_one_epoch(config, train_forward, model, loss_list, data_loader, optimizer, scaler, epoch, lr_scheduler, writer):
    torch.cuda.reset_peak_memory_stats()
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    losses_count = len(loss_list)
    losses_meter = [AverageMeter() for _ in range(losses_count)]

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs, targets = train_forward(config, model, data)

        losses = loss_list(outputs, targets)
        loss = sum(losses)

        optimizer.zero_grad()
        loss.backward()

        if config['train'].get('clip_grad'):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
        else:
            grad_norm = get_grad_norm(model.parameters())

        optimizer.step()

        if not config['train']['lr_scheduler']['t_in_epochs']:
            lr_scheduler.step_update((epoch-1)*num_steps+idx)

        batch_size = list(targets.values())[0].size(0)
        for _loss_meter, _loss in zip(losses_meter, losses):
            _loss_meter.update(_loss.item(), batch_size)   
        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config['print_per_iter'] == 0 or idx == num_steps:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config["train"]["epochs"]}][{idx}/{num_steps}]\t'
                f'ETA {datetime.timedelta(seconds=int(etas))} LR {lr:.6f}\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'Loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'
                f'GradNorm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB')
    if config['train']['lr_scheduler']['t_in_epochs']:
        lr_scheduler.step(epoch)
    logger.info(f"Train: [{epoch}/{config['train']['epochs']}] Time {datetime.timedelta(seconds=int(time.time()-start))}")
    tensor_board_dict = {'train/loss_total':loss_meter.avg}
    for index, (_loss, _loss_meter) in enumerate(zip(losses, losses_meter)):
        tensor_board_dict[f'train/loss_{index}'] = _loss_meter.avg
    for log_key, log_value in tensor_board_dict.items():
        writer.add_scalar(log_key, log_value, epoch)

@torch.no_grad()
def validate(config, test_forward, model, loss_list, data_loader, epoch, writer):
    torch.cuda.reset_max_memory_allocated()
    model.eval()

    logger.info(f"Valid: [{epoch}/{config['train']['epochs']}]\t")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    losses_count = len(loss_list)
    losses_meter = [AverageMeter() for _ in range(losses_count)]

    start = time.time()
    end = time.time()
    for idx, data in enumerate(data_loader):
        data_time.update(time.time() - end)
        outputs, targets, img_files, lbl_files = test_forward(config, model, data)

        if config['testset_as_validset']:
            psnr, ssim = test_metric_cuda(config, epoch, outputs[config['test']['which_stage']], targets[config['test']['which_gt']], img_files, lbl_files)
        else:
            psnr, ssim = validate_metric(config, epoch, outputs[config['test']['which_stage']], targets[config['test']['which_gt']], img_files, lbl_files)

        losses = loss_list(outputs, targets)
        loss = sum(losses)
        batch_size = targets[config['test']['which_gt']].size(0)
        for _loss_meter, _loss in zip(losses_meter, losses):
            _loss_meter.update(_loss.item(), batch_size)   
        loss_meter.update(loss.item(), batch_size)
        psnr_meter.update(psnr.item(), batch_size)
        ssim_meter.update(ssim.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if config['testset_as_validset'] or idx % config['print_per_iter'] == 0 or idx == len(data_loader):
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Valid: [{epoch}/{config["train"]["epochs"]}][{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                f'Loss {loss_meter.val:.8f} ({loss_meter.avg:.8f})\t'
                f'PSNR {psnr_meter.val:.4f} ({psnr_meter.avg:.4f})\t'
                f'SSIM {ssim_meter.val:.4f} ({ssim_meter.avg:.4f})\t'
                f'Mem {memory_used:.0f}MB\t{os.path.basename(img_files[0])}')
    logger.info(f'Valid: [{epoch}/{config["train"]["epochs"]}] PSNR {psnr_meter.avg:.4f}\tSSIM {ssim_meter.avg:.4f}')
    logger.info(f'Valid: [{epoch}/{config["train"]["epochs"]}] Time {datetime.timedelta(seconds=int(time.time()-start))}')
    tensor_board_dict = {'eval/loss_total':loss_meter.avg}
    for index, (loss, loss_meter) in enumerate(zip(losses, losses_meter)):
        tensor_board_dict[f'eval/loss_{index}'] = loss_meter.avg
    tensor_board_dict['eval/psnr'] = psnr_meter.avg
    tensor_board_dict['eval/ssim'] = ssim_meter.avg
    for log_key, log_value in tensor_board_dict.items():
        writer.add_scalar(log_key, log_value, epoch)
    return psnr_meter.avg, ssim_meter.avg, loss_meter.avg

@torch.no_grad()
def validate_metric(config, epoch, outputs, targets, image_paths, target_params=None):
    outputs = torch.clamp(outputs, 0, 1) * 255
    targets = targets * 255
    if config['test']['round']:
        outputs = outputs.round()
        targets = targets.round()
    psnrs = get_psnr_torch(outputs, targets)
    ssims = get_ssim_torch(outputs, targets)

    if config['test']['save_image'] and epoch % config['save_per_epoch'] == 0:
        images = torch.cat((outputs, targets), dim=3)
        result_path = os.path.join(config['output'], 'results', f'valid_{epoch:04d}')
        os.makedirs(result_path, exist_ok=True)
        for image, image_path, psnr in zip(images, image_paths, psnrs):
            save_path = os.path.join(result_path, f'{os.path.basename(image_path)[:-4]}_{psnr:.2f}.jpg')
            save_image_torch(image, save_path)

    return psnrs.mean(), ssims.mean()

@torch.no_grad()
def test_metric_cuda(config, epoch, outputs, targets, image_paths, target_params=None):
    outputs = torch.clamp(outputs, 0, 1) * 255
    targets = torch.clamp(targets, 0, 1) * 255
    if config['test']['round']:
        outputs = outputs.round()
        targets = targets.round()
    psnr = get_psnr_torch(outputs, targets)
    ssim = get_ssim_torch(outputs, targets)

    if config['test']['save_image']:
        result_path = os.path.join(config['output'], 'results', f'test_{epoch:04d}')
        os.makedirs(result_path, exist_ok=True)
        save_path = os.path.join(result_path, f'{os.path.basename(image_paths[0])[:-4]}_{psnr.item():.2f}.png')
        save_image_torch(outputs[0], save_path)

    return psnr, ssim

@torch.no_grad()
def throughput(config, forward, model, data_loader, logger):
    model.eval()

    for idx, data in enumerate(data_loader):
        for i in range(30):
            forward(config, model, data)
        logger.info(f"throughput averaged with 100 times")
        torch.cuda.synchronize()
        tic = time.time()
        for i in range(100):
            pred, label = forward(config, model, data)
        batch_size = list(pred.values())[0].size(0)
        torch.cuda.synchronize()
        toc = time.time()
        logger.info(f"batch_size {batch_size} throughput {(toc - tic) * 1000 / (100 * batch_size)}ms")
        return


if __name__ == '__main__':
    args, config = parse_options()
    phase = 'train' if not args.test else 'test'

    cudnn.benchmark = True

    os.makedirs(config['output'], exist_ok=True)
    start_time = time.strftime("%y%m%d-%H%M", time.localtime())
    logger = create_logger(output_dir=config['output'], name=f"{config['tag']}", action=f"{phase}-{start_time}")
    path = os.path.join(config['output'], f"{phase}-{start_time}.yaml")
    
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        if repo.is_dirty():
            logger.warning(f'Current work on commit: {sha}, however the repo is dirty (not committed)!')
        else:
            logger.info(f'Current work on commit: {sha}.')
    except git.exc.InvalidGitRepositoryError as e:
        logger.warn(f'No git repo base.')

    copy_cfg(config, path)
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info("Config:\n" + yaml.dump(ordered_dict_to_dict(config), default_flow_style=False, sort_keys=False))
    current_cuda_device = torch.cuda.get_device_properties(torch.cuda.current_device())
    logger.info(f"Current CUDA Device: {current_cuda_device.name}, Total Mem: {int(current_cuda_device.total_memory / 1024 / 1024)}MB")

    main(config)
