import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device

from Attention import Attention


class UV_Aggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda=None, uv=True):
        super(UV_Aggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device =  torch.device(cuda if cuda is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, history_uv, history_r):
        """
        nodes: list of nodes
        history_uv: list of list, node's history interacted items
        history_r: list of list, node's history interacted ratings
        """
        # Don't convert history_uv and history_r to tensors directly
        # They are lists of lists with varying lengths

        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(nodes)):
            history = history_uv[i]
            num_history_item = len(history)

            if num_history_item == 0:
                # Handle the case where a node has no history
                if self.uv:
                    embed_matrix[i] = self.u2e.weight[nodes[i]]
                else:
                    embed_matrix[i] = self.v2e.weight[nodes[i]]
                continue

            tmp_label = history_r[i]

            # Convert history and labels to tensors for this specific node
            history_tensor = torch.LongTensor(history).to(self.device)
            label_tensor = torch.LongTensor(tmp_label).to(self.device)

            if self.uv:
                # user component
                e_uv = self.v2e.weight[history_tensor]
                uv_rep = self.u2e.weight[nodes[i]].expand(num_history_item, -1)
            else:
                # item component
                e_uv = self.u2e.weight[history_tensor]
                uv_rep = self.v2e.weight[nodes[i]].expand(num_history_item, -1)

            e_r = self.r2e.weight[label_tensor]

            x = torch.cat((e_uv, e_r), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            # Calculate attention weights
            att_w = self.att(o_history, uv_rep[0], num_history_item)

            # Apply attention weights
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history

        return embed_matrix