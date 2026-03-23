import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import torch
import torch.optim as optim
import pprint as pp
import random
import statistics

from utils.process import get_options
from tensorboard_logger import Logger as TbLogger
from learning.attention_model import AttentionModel
from learning.cgp import GenomeFactory, Genome, GenomeNN
from learning.encoders.graph_encoder import GraphAttentionEncoder
from learning.reinforce_baselines import ExponentialBaseline, RolloutBaseline, NoBaseline, WarmupBaseline
from learning.problem_vrp import CVRP
from utils.training import train_epoch
from utils.metrics import MetricsLogger


def calculate_scores(opts, tb_logger, encoder: GenomeNN, osobnik_id):
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
    val_dataset = CVRP.make_dataset(size=opts.graph_size, num_samples=opts.val_size)
    scores_per_epochs = []
    for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
        score = train_epoch(
            model,
            optimizer,
            baseline,
            lr_scheduler,
            epoch,
            val_dataset,
            tb_logger,
            opts
        )
        # metrics_logger.record(
        #         epoch=epoch,
        #         osobnik=osobnik_id,
        #         score=score
        # )
        scores_per_epochs.append(score)
        print(f'epoka {epoch}, genom {osobnik_id}, score {score}')
    return scores_per_epochs

def run_cgp(opts, tb_logger):
    genomes : list[Genome] = [] 
    baseline_genom = [(4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4), (1,)]*3
    genomes.append(Genome(baseline_genom))

    logger = MetricsLogger()

    for i in range(10):
        genome = GenomeEncoder.spawn_random_genome()
        genomes.append(genome)

    generations = 100
    for gen in range(generations):
        for i, genome in enumerate(genomes):
            if genome.score == None:
                scores = calculate_scores(opts, tb_logger, genome, logger, osobnik_id=genome.id)
                genome.score = scores[4] - 5 * ((scores[4] - scores[0] )/ 4)**2
                logger.record(
                        osobnik=genome.id,
                        genes=genome.genes,
                        score=genome.score
                )
        logger.save_csv("evolution1.csv")
        genomes.sort(key=lambda g: g.score)
        genomes = genomes[:6]
        print(f'Generacja {gen}. Najlepszy wynik: {scores[0]:.3f}. Średnia populacji: {torch.mean(torch.stack(scores)):.3f}')

        genomes.extend(crossover(genomes[0], genomes[1]))
        genomes.extend(crossover(genomes[2], genomes[3]))
        genomes.extend(crossover(genomes[4], genomes[5]))
        genomes.append(mutate(genomes[0]))
        genomes.append(mutate(genomes[1]))
        genomes.append(mutate(genomes[2]))
        genomes.append(mutate(genomes[3]))
        genomes.append(mutate(genomes[4]))
        genomes.append(mutate(genomes[5]))
        genomes.append(GenomeEncoder.spawn_random_genome())
        genomes.append(GenomeEncoder.spawn_random_genome())
    

def run(opts):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    #pp.pprint(vars(opts))
    torch.manual_seed(opts.seed)
    random.seed(opts.seed)
    tb_logger = TbLogger(os.path.join(opts.log_dir, "VRP_{}".format(opts.graph_size), opts.run_name))
    os.makedirs(opts.save_dir)
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    opts.n_epochs = 5
    opts.graph_size = 10
    opts.val_size = 10000
    opts.epoch_size = 128000
    #opts.epoch_size = 1280
    opts.n_epochs = 1
    opts.epoch_size = 12800

    factory = GenomeFactory()
    baseline = factory.produce_genome([ (4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4),  (1,)]*3)
    random.seed(opts.seed)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization
    )

    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    mine_encoder = baseline.build_nn(opts)
    scores = calculate_scores(opts, tb_logger, mine_encoder, osobnik_id=0)
    print("")
    random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    original_encoder = GraphAttentionEncoder(
        n_heads=opts.n_heads,
        embed_dim=opts.embedding_dim,
        n_layers=opts.n_encode_layers,
        normalization=opts.normalization
    )
    scores = calculate_scores(opts, tb_logger, original_encoder, osobnik_id=1)

    #print(scores)
    #opts.n_epochs = 100
    # opts.graph_size = 10
    # print(f"EPOCHS: {opts.n_epochs}")
    # print(f"GRAPH SIZE: {opts.graph_size}")
    # scores = []
    # genomes = []
    # baseline_genom = GenomeEncoder(opts.emb, [(4,), (5, -2), (1,), (6, 1), (7,), (6, -1), (5, -4), (1,)]*3)
    # genomes.append(baseline_genom)
    # scores = calculate_scores(opts, tb_logger, genomes[0], osobnik_id=0)
    # print(scores)
    # for i in range(4):
    #     genome = GenomeEncoder.spawn_random_genome()
    #     genomes.append(genome)

    #score = calculate_score(opts, tb_logger, genome, logger, osobnik_id=i)
    # for i, genome in enumerate(genomes):
    #     score = calculate_score(opts, tb_logger, genome, logger, osobnik_id=i)

    # logger.save_csv("vrp10_20_epochs_epoch_128000.csv")



if __name__ == "__main__":
    run(get_options())
 