import torch
import torch.nn as nn
from .Attention import Attention
from torch import device


class Social_Aggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, features, u2e, embed_dim, cuda=device):
        super(Social_Aggregator, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]
            num_neighs = len(tmp_adj)

            if num_neighs == 0:
                # Option 1: Use the node's own embedding as fallback
                embed_matrix[i] = self.u2e.weight[nodes[i]]
                continue

            # Fast user embedding
            e_u = self.u2e.weight[list(tmp_adj)]

            # User representation
            u_rep = self.u2e.weight[nodes[i]]

            # Attention weights
            att_w = self.att(e_u, u_rep, num_neighs)

            if att_w.dim() == 1:
                att_w = att_w.view(-1, 1)

            # Matrix multiplication
            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history

        return embed_matrix

