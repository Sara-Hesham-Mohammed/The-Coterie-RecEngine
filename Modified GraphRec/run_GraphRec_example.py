import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

Modified to include LSTM components for temporal sequence modeling.
"""


class SequenceAggregator(nn.Module):
    """
    Aggregates a user's or item's interaction history using LSTM
    """

    def __init__(self, embed_dim, hidden_dim, num_layers=1, dropout=0.2, cuda="cpu"):
        super(SequenceAggregator, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.device = cuda

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, seq_embeds, seq_lengths):
        """
        Args:
            seq_embeds: Batch of sequence embeddings [batch_size, max_seq_len, embed_dim]
            seq_lengths: Length of each sequence in the batch [batch_size]
        """
        batch_size = seq_embeds.size(0)

        # Pack padded sequence for efficient computation
        packed_input = nn.utils.rnn.pack_padded_sequence(
            seq_embeds, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process through LSTM
        packed_output, (hidden, _) = self.lstm(packed_input)

        # Unpack output
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Attention mechanism over LSTM outputs
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Apply attention weights
        weighted_output = torch.sum(lstm_out * attn_weights, dim=1)

        return weighted_output


class GraphRecLSTM(nn.Module):
    def __init__(self, enc_u, enc_v_history, r2e, hidden_dim, history_u_lists, history_v_lists, cuda="cpu"):
        super(GraphRecLSTM, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim
        self.device = cuda

        # LSTM components for sequential modeling
        self.user_seq_aggregator = SequenceAggregator(self.embed_dim, hidden_dim, cuda=cuda)
        self.item_seq_aggregator = SequenceAggregator(self.embed_dim, hidden_dim, cuda=cuda)

        # Original components
        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)

        # Modified to incorporate LSTM outputs
        self.w_uv1 = nn.Linear(self.embed_dim * 2 + hidden_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)

        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

        # Store user and item histories for sequence generation
        self.history_u_lists = history_u_lists
        self.history_v_lists = history_v_lists

    def prepare_sequences(self, nodes_u, nodes_v, u2e, v2e, r2e, history_u_lists, history_ur_lists, history_v_lists,
                          history_vr_lists, max_seq_len=10):
        """
        Prepare sequential data for users and items
        """
        batch_size = len(nodes_u)

        # Initialize user sequence data
        user_seq_embeds = torch.zeros(batch_size, max_seq_len, self.embed_dim).to(self.device)
        user_seq_lengths = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        # Initialize item sequence data
        item_seq_embeds = torch.zeros(batch_size, max_seq_len, self.embed_dim).to(self.device)
        item_seq_lengths = torch.zeros(batch_size, dtype=torch.long).to(self.device)

        # Process user sequences
        for i, user in enumerate(nodes_u):
            user_id = user.item()
            if user_id in history_u_lists:
                # Get user's item interaction history
                items = list(history_u_lists[user_id])
                ratings = [history_ur_lists[user_id][item] for item in items]

                # Limit sequence length
                seq_len = min(len(items), max_seq_len)
                user_seq_lengths[i] = seq_len

                # Most recent interactions first
                items = items[-seq_len:]
                ratings = ratings[-seq_len:]

                # Create embeddings for each (item, rating) pair
                for j, (item, rating) in enumerate(zip(items, ratings)):
                    item_embed = v2e(torch.LongTensor([item]).to(self.device)).squeeze()
                    rating_embed = r2e(torch.LongTensor([rating]).to(self.device)).squeeze()

                    # Combine item and rating embeddings
                    interaction_embed = item_embed * rating_embed
                    user_seq_embeds[i, j] = interaction_embed
            else:
                user_seq_lengths[i] = 1
                user_seq_embeds[i, 0] = torch.zeros(self.embed_dim).to(self.device)

        # Process item sequences
        for i, item in enumerate(nodes_v):
            item_id = item.item()
            if item_id in history_v_lists:
                # Get item's user interaction history
                users = list(history_v_lists[item_id])
                ratings = [history_vr_lists[item_id][user] for user in users]

                # Limit sequence length
                seq_len = min(len(users), max_seq_len)
                item_seq_lengths[i] = seq_len

                # Most recent interactions first
                users = users[-seq_len:]
                ratings = ratings[-seq_len:]

                # Create embeddings for each (user, rating) pair
                for j, (user, rating) in enumerate(zip(users, ratings)):
                    user_embed = u2e(torch.LongTensor([user]).to(self.device)).squeeze()
                    rating_embed = r2e(torch.LongTensor([rating]).to(self.device)).squeeze()

                    # Combine user and rating embeddings
                    interaction_embed = user_embed * rating_embed
                    item_seq_embeds[i, j] = interaction_embed
            else:
                item_seq_lengths[i] = 1
                item_seq_embeds[i, 0] = torch.zeros(self.embed_dim).to(self.device)

        return user_seq_embeds, user_seq_lengths, item_seq_embeds, item_seq_lengths

    def forward(self, nodes_u, nodes_v, history_u_lists=None, history_ur_lists=None, history_v_lists=None,
                history_vr_lists=None):
        # If history lists are not provided, use the stored ones
        if history_u_lists is None:
            history_u_lists = self.history_u_lists
        if history_v_lists is None:
            history_v_lists = self.history_v_lists

        # Get graph embeddings from original GraphRec
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        # Prepare sequential data
        u2e = self.enc_u.embedding
        v2e = self.enc_v_history.embedding
        user_seq_embeds, user_seq_lengths, item_seq_embeds, item_seq_lengths = self.prepare_sequences(
            nodes_u, nodes_v, u2e, v2e, self.r2e,
            history_u_lists, history_ur_lists, history_v_lists, history_vr_lists
        )

        # Process sequences through LSTM
        user_seq_output = self.user_seq_aggregator(user_seq_embeds, user_seq_lengths)
        item_seq_output = self.item_seq_aggregator(item_seq_embeds, item_seq_lengths)

        # Process graph embeddings (original GraphRec)
        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)

        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        # Combine graph embeddings with LSTM outputs
        x_uv = torch.cat((x_u, x_v, user_seq_output, item_seq_output), 1)

        # Final prediction layers
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)

        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list, history_u_lists=None, history_ur_lists=None, history_v_lists=None,
             history_vr_lists=None):
        scores = self.forward(nodes_u, nodes_v, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae, history_u_lists, history_ur_lists,
          history_v_lists, history_vr_lists):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(
            batch_nodes_u.to(device),
            batch_nodes_v.to(device),
            labels_list.to(device),
            history_u_lists,
            history_ur_lists,
            history_v_lists,
            history_vr_lists
        )
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(
                test_u,
                test_v,
                history_u_lists,
                history_ur_lists,
                history_v_lists,
                history_vr_lists
            )
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model with LSTM')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--hidden_dim', type=int, default=128, metavar='N', help='LSTM hidden size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    dir_data = './data/toy_dataset'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
        data_file)
    """
    ## toy dataset 
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)
    r2e = nn.Embedding(num_ratings, embed_dim).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                           base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # model - now using GraphRecLSTM instead of GraphRec
    graphrec = GraphRecLSTM(enc_u, enc_v_history, r2e, hidden_dim, history_u_lists, history_v_lists, cuda=device).to(
        device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):
        train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae,
              history_u_lists, history_ur_lists, history_v_lists, history_vr_lists)
        expected_rmse, mae = test(graphrec, device, test_loader,
                                  history_u_lists, history_ur_lists, history_v_lists, history_vr_lists)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break


if __name__ == "__main__":
    main()