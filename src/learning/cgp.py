import math
import torch
import random
import time 
from collections import defaultdict, deque
from dataclasses import dataclass

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
    def __init__(self, input_dims, embed_dim):
        super(Add, self).__init__()
        self.embed_dim = embed_dim

        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim)
            if dim != embed_dim
            else nn.Identity()
            for dim in input_dims
        ])

    def forward(self, xs):
        projected = [ proj(x) for x, proj in zip(xs, self.projections) ]
        return sum(projected)
    # def forward(self, xs):
    #     assert len(xs) >= 2, "Add requires at least 2 tensors"

    #     base_shape = xs[0].shape

    #     for i, x in enumerate(xs[1:], 1):
    #         assert x.shape == base_shape, (
    #             f"Shape mismatch in Add: "
    #             f"xs[0]={base_shape}, xs[{i}]={x.shape}"
    #         )

    #     out = xs[0]
    #     #print(f"ADDING = {out.flatten()[0].item():.4f}")
    #     for x in xs[1:]:
    #         #print(f"ADDING = {x.flatten()[0].item():.4f}")
    #         out = out + x

    #     return out
@dataclass
class Gene:
    pos: int
    type: int
    inputs: list[int]
    args: list[int]
    
@dataclass
class CGP_Element:
    nn: nn.Module
    dim: int
    active: bool = False

def parse_gene(t, pos):
    if not t:
        return None
    type_ = t[0]
    rest = list(t[1:])
    if not rest:
        raise ValueError(f"Gene {t} has no inputs")
    first = rest[0]
    if isinstance(first, tuple):
        inputs = list(first)
        args = rest[1:]
    else:
        inputs = [first]
        args = rest[1:]
    return Gene(pos, type_, inputs, args)

def parse_genes(data):
    return [parse_gene(t, idx) for idx, t in enumerate(data)]

# 1 - Identity
# 2 - Normalization
# 3 - Linear scaling
# 4 - MultiHeadAttention
# 5 - Add
# 6 - Gelu
# 7 - Relu
GENE_TYPES_LEN = 7

# (TYP, (INPUTY), (PARAMS))
class CGP_Net(nn.Module):
    def __init__(self, embed_dim, x_dim, y_dim, outputs, 
                 genes=None, nets=None, genome=None):
        super().__init__()
        self.num_heads = 8
        self.feed_forward_hidden = 512
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.len = self.x_dim * self.y_dim + 2
        self.embed_dim = embed_dim
        self.outputs = outputs
        if genome:
            self.genes = parse_genes([None, *genome, None])
        elif genes:
            self.genes = genes
        else:
            self.genes = [None]*self.len
        self.genes[self.len - 1] = parse_gene((5, tuple(outputs)), self.len-1)
        if nets:
            self.nets = nets
            for pos, net in enumerate(self.nets):
                if net is not None and net.nn is not None:
                    self.add_module(f"node_{pos}", net.nn)
        else:
            self.nets = [CGP_Element(None, self.embed_dim), *[None] * (self.len-1)]
            for x in range(self.x_dim):
                for y in range(self.y_dim):
                    idx = self.to_global_idx(x, y)
                    if not self.genes[idx]:
                        continue
                    self.nets[idx] = self.produce_gene(self.genes[idx])
            self.nets[self.len - 1] = self.produce_gene(self.genes[self.len - 1])
        assert len(self.genes) == self.len
        assert len(self.nets) == self.len
        self.mark_active_paths()
        self.build_propagation_order()


    @classmethod
    def from_parent(cls, genes, nets, outputs, parent):
        return CGP_Net(parent.embed_dim, parent.x_dim, parent.y_dim, outputs,
                       genes=genes, nets=nets)

    def forward(self, x):
        outputs = [None]*self.len
        outputs[0] = x    
        
        # print("\n========== CGP FORWARD ==========")
        # print(
        #     f"[INPUT] "
        #     f"shape={x.shape} "
        #     f"mean={x.mean().item():.4f} "
        #     f"std={x.std().item():.4f} "
        #     f"first={x.flatten()[0].item():.4f}"
        # )
        for idx in self.propagation_order:
            if idx == 0:
                continue
            inputs = self.genes[idx].inputs
            in_vals = [outputs[i] for i in inputs]
            if self.genes[idx].type == 5:
                outputs[idx] = self.nets[idx].nn(in_vals)
                #print(f'pos: {idx}, inputs: {self.genes[idx].inputs}')
            elif len(in_vals) == 1:
                outputs[idx] = self.nets[idx].nn(in_vals[0])
            else:
                outputs[idx] = self.nets[idx].nn(in_vals)
            out = outputs[idx] 
            # if isinstance(out, torch.Tensor):
            #     print(
            #         f"TYP {self.genes[idx].type} output "
            #         f"shape={out.shape} "
            #         f"mean={out.mean().item():.4f} "
            #         f"std={out.std().item():.4f} "
            #         f"first={out.flatten()[0].item():.4f}"
            #     )

            #     if torch.isnan(out).any():
            #         print(" !!! NAN DETECTED !!! ")

            # else:
            #     print(f" output type={type(out)}")

        # outputs[self.len-1] = outputs[47]
        final_h = outputs[self.len-1]
        graph_embedding = final_h.mean(dim=1)
        
        # print("\n========== FINAL ==========")

        # print(
        #     f"final_h "
        #     f"shape={final_h.shape} "
        #     f"mean={final_h.mean().item():.4f} "
        #     f"std={final_h.std().item():.4f} "
        # )

        graph_embedding = final_h.mean(dim=1)

        # print(
        #     f"graph_embedding "
        #     f"shape={graph_embedding.shape} "
        #     f"mean={graph_embedding.mean().item():.4f} "
        #     f"std={graph_embedding.std().item():.4f} "
        #     f"norm={graph_embedding.norm().item():.4f}"
        # )

        # print("=================================\n")
        return (final_h, graph_embedding)

    def mark_active_paths(self):
        visited = set()
        queue = deque()
        queue.append(self.len - 1)

        while queue:
            pos = queue.popleft()
            if pos == 0 or pos in visited:
                continue
            self.nets[pos].active  = True
            visited.add(pos)
            queue.extend(self.genes[pos].inputs)

    # def build_propagation_order(self):
    #     graph = defaultdict(list)
    #     indeg = defaultdict(int)

    #     # budujemy dependency graph
    #     for i, g in enumerate(self.genes):
    #         if g is None:
    #             continue
    #         for inp in g.inputs:
    #             graph[inp].append(i)
    #             indeg[i] += 1

    #     # source node
    #     q = deque([0])
    #     order = []

    #     while q:
    #         u = q.popleft()
    #         order.append(u)

    #         for v in graph[u]:
    #             indeg[v] -= 1
    #             if indeg[v] == 0:
    #                 q.append(v)

    #     self.propagation_order = order
    def build_propagation_order(self):
        order =[]
        for x in range(self.x_dim):
            for y in range(self.y_dim): 
                idx = self.to_global_idx(x, y)
                if self.nets[idx] and self.nets[idx].active:
                    order.append(idx)
        order.append(self.len - 1)
        self.propagation_order = order
        
    def to_global_idx(self, x, y):
        return y*self.x_dim + x + 1

    def to_xy(self, pos):
        x = (pos - 1) // self.x_dim
        y = (pos / 1 ) % self.x_dim
        return (x, y)

    def produce_gene(self, gene: Gene):
        first_input_dim = self.embed_dim
        if gene.inputs[0] == 0:
            first_input_dim = self.embed_dim
        else:
            first_input_dim = self.nets[gene.inputs[0]].dim
        output_dim = first_input_dim
        if gene.type == 1:
            net = nn.Identity()
        elif gene.type == 2:
            net = Normalization(first_input_dim)
        elif gene.type == 3:
            scaling = gene.args[0]
            if scaling == 1 and first_input_dim <= 2048:
                output_dim = first_input_dim * 4
                net = nn.Linear(first_input_dim, output_dim)
            elif scaling == -1 and first_input_dim >= 8:
                output_dim = first_input_dim // 4
                net = nn.Linear(first_input_dim, output_dim)
            else:
                net = nn.Linear(first_input_dim, first_input_dim)
        elif gene.type == 4:
            net = MultiHeadAttention(self.num_heads, first_input_dim, first_input_dim)
        elif gene.type == 5:
            if len(gene.inputs) == 1:
                gene.type = 1
                net = nn.Identity()
            else:
                input_dims = [
                    self.nets[i].dim
                    for i in gene.inputs
                ]
                net = Add(input_dims, self.embed_dim)
                output_dim = self.embed_dim
        elif gene.type == 6:
            net = nn.GELU()
        elif gene.type == 7:
            net = nn.ReLU()
        elif gene.type == 8:
            net = nn.LayerNorm()
        else:
            raise f'unknown layer type {gene.type}'
        
        self.add_module(f"node_{gene.pos}", net)
        return CGP_Element(net, output_dim)

    def produce_offspring(self, n, p_mut=0.1):
        children = []
        for i in range(n):
            genes = [None]*self.len
            nets = [None]*self.len
            nets[0] = self.nets[0]
            for pos in range(1, self.len):
                x,y = self.to_xy(pos)
                if False and random.random() < p_mut:
                    mutation_type = random.choice(["type", "params"])
                    if mutation_type == "type":
                        new_type = random.randint(0, GENE_TYPES_LEN)
                else:
                    nets[pos] = self.nets[pos]
                    genes[pos] = self.genes[pos]
            children.append(CGP_Net.from_parent(genes, nets, self.outputs, self))
        return children
    
    def save_snapshot(self):
        snapshot = {}
        for pos, net_element in enumerate(self.nets):
            if net_element and net_element.nn:
                snapshot[f"net_{pos}"] = net_element.nn.state_dict()
        return snapshot

    def load_snapshot(self, snapshot):
        for pos, net_element in enumerate(self.nets):
            if net_element and net_element.nn and f"net_{pos}" in snapshot:
                net_element.nn.load_state_dict(snapshot[f"net_{pos}"])
            
    def mutate_gene(self, gene, i):
        layer_type = gene[0]
        mutation_type = random.choice(["type", "param"])

        # zmiana typu
        if mutation_type == "type":
            return self.spawn_gene(i)

        # zmiana parametru
        else:
            if layer_type == 5:
                skip_from = skip_from = random.randint(1, i + 1)
                return (5, -skip_from)
            elif layer_type == 6:
                return (6, random.randint(-1, 1))
            else:
                return self.spawn_gene(i)
    