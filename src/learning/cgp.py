import math
import torch
import random

from torch import nn

class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out

class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input
    
class Add(nn.Module):
    """Simple adding for making skip connection possible"""
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, skip):
        return x + skip


# 1 - normalization 
# 2 - attention block
# 3 - feedforward block
# 4 - multihead attention
# 5 - add ( X - skip z X kroków wcześniej)
# 6 - Linear (1 - UP DIMEN, 2 - DOWN DIMEN)
# 7 - RELU
# genome [(1,2),  (1,3)]
class GenomeEncoder(nn.Module):
    def __init__(self, embed_dim, genome):
        super().__init__()
        self.num_heads = 8
        self.feed_forward_hidden = 512
        self.genome = genome
        self.layers = nn.ModuleList()

        for gene in self.genome:
            layer_type = gene[0]
            if layer_type == 1:
                layer = Normalization(embed_dim)
            elif layer_type == 2:
                layer = SkipConnection(
                    MultiHeadAttention(
                        self.num_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim
                    )
                )
            elif layer_type == 3:
                layer =  SkipConnection(
                    nn.Sequential(
                        nn.Linear(embed_dim, self.feed_forward_hidden ),
                        nn.ReLU(),
                        nn.Linear(self.feed_forward_hidden , embed_dim)
                    )
                )
            elif layer_type == 4:
                layer = MultiHeadAttention(
                    self.num_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            elif layer_type == 5:
                layer = Add()
            elif layer_type == 6:
                scaling = gene[1]
                if scaling == 1: # scale up
                    layer = nn.Linear(embed_dim, self.feed_forward_hidden)
                elif scaling == -1: # scale down
                    layer = nn.Linear(self.feed_forward_hidden, embed_dim)    
            elif layer_type == 7:
                layer = nn.ReLU()
            self.layers.append(layer)

    def forward(self, x, mask=None):
        """
        x shape:
        (batch_size, seq_len, embed_dim)
        """

        outputs = [x]

        for i, gene in enumerate(self.genome):
            layer_type = gene[0]

            layer = self.layers[i]
            inp = outputs[-1]

            if layer_type == 5:
                skip_rel = gene[1] + 1
                skip_idx = i + skip_rel
                out = layer(inp, outputs[skip_idx])
            else:
                out = layer(inp)

            outputs.append(out)

        final_h = outputs[-1]
        graph_embedding = final_h.mean(dim=1)
        return (final_h, graph_embedding)

    @staticmethod
    def spawn_random_genome():
        mu, sigma = 7, 2  # średnia 7, odchylenie standardowe 2
        length = int(torch.normal(mean=mu, std=sigma).item())
        length = max(1, length)  # minimalna długość 1

        genome = []
        for i in range(length):
            layer_type = random.randint(1, 7)
            
            