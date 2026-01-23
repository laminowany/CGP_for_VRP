import os
import json
import torch
import torch.optim as optim

from utils.process import get_options
from tensorboard_logger import Logger as TbLogger
from learning.base_net import AttentionModel
from learning.reinforce_baselines import ExponentialBaseline, RolloutBaseline, NoBaseline, WarmupBaseline

def run(opts):
    torch.manual_seed(opts.seed)
    tb_logger = TbLogger(os.path.join(opts.log_dir, "VRP_{}".format(opts.problem_size), opts.run_name))
    
    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")


    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)

     # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

        # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    
    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)


if __name__ == "__main__":
    run(get_options())
 