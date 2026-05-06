import time
import torch
import math
import torch.optim as optim
import random

from tqdm import tqdm
from torch.utils.data import DataLoader
from learning.attention_model import AttentionModel
from learning.cgp import Genome
from learning.reinforce_baselines import RolloutBaseline, WarmupBaseline
from utils.logger import Logger
from utils.misc import move_to
from learning.problem_vrp import CVRP


def evaluate(opts, genome: Genome, logger: Logger, osobnik_id = None):
    encoder = genome.build_nn(opts)
    try:
        score = evaluate_with_encoder(opts, encoder, logger=logger, osobnik_id=osobnik_id)
    except Exception as e:
        score = None
        print(f"Exception while evaluating {genome.genes}")
        print(e)
    return score

def evaluate_with_encoder(opts, encoder, logger: Logger, osobnik_id = None):
    if not osobnik_id:
        osobnik_id = 0

    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        encoder=encoder,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

    baseline = RolloutBaseline(model, opts)
    baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)
    validation_set = getattr(opts, "validation_set", None)
    if not validation_set:
        validation_set = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    
    if opts.reproducible_seed:
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)

    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        start = time.perf_counter()
        try:
            score = train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                validation_set,
                opts
            )
        except TimeoutError:
            return None
        end = time.perf_counter()
        logger.record(
                epoch=epoch,
                osobnik=osobnik_id,
                score=score,
                time=end-start
        )
    return score

def validate(model, dataset, opts):
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
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
    start = time.perf_counter()
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            batch,
            opts
        )
        execution_time = time.perf_counter() - start
        if execution_time > opts.epoch_time_limit:
            raise TimeoutError
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

