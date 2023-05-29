from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

def build_scheduler(config, optimizer, n_iter_per_epoch=1):
    if config['lr_scheduler']['t_in_epochs']:
        n_iter_per_epoch = 1
    num_steps = int(config['epochs'] * n_iter_per_epoch)
    warmup_steps = int(config['warmup_epochs'] * n_iter_per_epoch)
    lr_scheduler = None
    if config['lr_scheduler']['type'] == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.,
            lr_min=config['min_lr'],
            warmup_lr_init=config.get('warmup_lr', 0.0),
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=config['lr_scheduler']['t_in_epochs'],
        )
    elif config['lr_scheduler']['type'] == 'step':
        decay_steps = int(config['lr_scheduler']['decay_epochs'] * n_iter_per_epoch)
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config['lr_scheduler']['decay_rate'],
            warmup_lr_init=config.get('warmup_lr', 0.0),
            warmup_t=warmup_steps,
            t_in_epochs=config['lr_scheduler']['t_in_epochs'],
        )
    else:
        raise NotImplementedError()

    return lr_scheduler
