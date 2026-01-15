import os
import json
import torch

from utils.process import get_options
from tensorboard_logger import Logger as TbLogger
from nets.base_net import AttentionModel


def run(opts):
    torch.manual_seed(opts.seed)
    tb_logger = TbLogger(os.path.join(opts.log_dir, "VRP_{}".format(opts.problem_size), opts.run_name))
    
    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)
    opts.device = torch.device("cuda:0")


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


if __name__ == "__main__":
    run(get_options())
 