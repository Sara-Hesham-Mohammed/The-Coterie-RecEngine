import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, node1, node2, num_neighs):
        if num_neighs == 0:
            # Handle the case where there are no neighbors
            return torch.zeros(1).to(node1.device)

        # If node2 is already expanded to match node1 shape, use it directly
        if len(node2.shape) > 1 and node2.shape[0] == node1.shape[0]:
            uv_reps = node2
        else:
            # Otherwise expand node2 to match node1's shape
            uv_reps = node2.expand(num_neighs, -1)

        x = torch.cat((node1, uv_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)

        att = self.softmax(x)
        return att