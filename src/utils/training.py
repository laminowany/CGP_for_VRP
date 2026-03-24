import time
import os
import torch
import math

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.misc import move_to, log_values
from learning.problem_vrp import CVRP


def validate(model, dataset, opts):
    # Validate
    #print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    # print('Validation overall avg_cost: {} +- {}'.format(
    #     avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    model.set_decode_type("greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, opts):
    #print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(CVRP.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size))
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    # Put model in train mode!
    model.train()
    model.set_decode_type("sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            batch,
            opts
        )

        step += 1

    #epoch_duration = time.time() - start_time
    #print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # if epoch == opts.n_epochs - 1:
    #     dest =  os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    #     print(f'Saving model and state... to {dest}')
    #     torch.save(
    #     {
    #         'model': model,
    #         'optimizer': optimizer.state_dict(),
    #         'rng_state': torch.get_rng_state(),
    #         'cuda_rng_state': torch.cuda.get_rng_state_all(),
    #         'baseline': baseline.state_dict()
    #     },
    #     os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    # )

    avg_reward = validate(model, val_dataset, opts)
    baseline.epoch_callback(model, epoch)
    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    return avg_reward



def train_batch(
        model,
        optimizer,
        baseline,
        batch,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood = model(x)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

