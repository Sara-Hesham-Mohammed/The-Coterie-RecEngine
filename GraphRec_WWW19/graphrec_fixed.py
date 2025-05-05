import torch
import torch.nn as nn
import pickle
import numpy as np
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]
"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 1)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def get_recommended_events(model, user_id, all_events, device, top_k=10, history_u_lists=None):
    """
    Get recommended events for a user

    Args:
        model: Trained GraphRec model
        user_id: User ID to get recommendations for
        all_events: List of all event IDs
        device: Device to run inference on (cuda/cpu)
        top_k: Number of recommendations to return
        history_u_lists: Dictionary of user's event history

    Returns:
        List of tuples (event_id, predicted_rating)
    """
    model.eval()

    # Get events the user hasn't interacted with
    if history_u_lists and user_id in history_u_lists:
        user_events = set(history_u_lists[user_id])
    else:
        user_events = set()  # New user with no history

    # Filter to only events the user hasn't seen
    candidate_events = [e for e in all_events if e not in user_events]

    # Prepare batches for prediction
    batch_size = 128
    predictions = []

    with torch.no_grad():
        # Process in batches to avoid memory issues
        for i in range(0, len(candidate_events), batch_size):
            batch_events = candidate_events[i:i + batch_size]

            # Create tensors
            test_usr = torch.full((len(batch_events),), user_id, dtype=torch.long).to(device)
            test_item = torch.tensor(batch_events, dtype=torch.long).to(device)

            # Get predictions
            pred_ratings = model.forward(test_usr, test_item)

            # Store results
            for j, event_id in enumerate(batch_events):
                predictions.append((event_id, pred_ratings[j].item()))

    # Sort by predicted rating
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Return top-k recommendations
    return sorted_predictions[:top_k]


def save_model_with_metadata(model, model_path, data_path, embed_dim):
    """
    Save model with metadata for easy loading

    Args:
        model: Trained GraphRec model
        model_path: Path to save model
        data_path: Path to the data pickle file
        embed_dim: Embedding dimension used
    """
    metadata = {
        'model_state': model.state_dict(),
        'data_path': data_path,
        'embed_dim': embed_dim
    }
    torch.save(metadata, model_path)
    print(f"Model saved to {model_path} with metadata")


def load_model_for_inference(model_path, device):
    """
    Returns:
        model: Loaded GraphRec model
        data_path: Path to the data pickle file
        embed_dim: Embedding dimension used
    """
    metadata = torch.load(model_path, map_location=device)

    # Get metadata
    data_path = metadata['data_path']
    embed_dim = metadata['embed_dim']

    # Load data
    with open(data_path, 'rb') as data_file:
        history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, _, _, _, _, _, _, social_adj_lists, ratings_list = pickle.load(
            data_file)

    # Setup model components
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

    # Create model
    model = GraphRec(enc_u, enc_v_history, r2e).to(device)

    # Load state
    model.load_state_dict(metadata['model_state'])
    model.eval()

    print(f"Model loaded from {model_path}")
    return model, history_u_lists, history_v_lists, ratings_list


def add_new_user(user_id, social_connections, history_u_lists, history_ur_lists, social_adj_lists):
    """
    Add a new user to the system

    Args:
        user_id: New user ID (should be the next available ID)
        social_connections: List of existing user IDs this user is connected to
        history_u_lists: Dictionary of user's event history to update
        history_ur_lists: Dictionary of user's rating history to update
        social_adj_lists: Dictionary of social connections to update

    Returns:
        Updated history_u_lists, history_ur_lists, social_adj_lists
    """
    # Initialize empty histories for the new user
    history_u_lists[user_id] = []
    history_ur_lists[user_id] = []

    # Add social connections
    social_adj_lists[user_id] = set(social_connections)

    # Add this user to their connections' social lists (bidirectional)
    for connection in social_connections:
        if connection in social_adj_lists:
            social_adj_lists[connection].add(user_id)

    return history_u_lists, history_ur_lists, social_adj_lists


def update_user_event_interaction(user_id, event_id, rating, history_u_lists, history_ur_lists,
                                  history_v_lists, history_vr_lists):
    """
    Update user-event interaction

    Args:
        user_id: User ID
        event_id: Event ID
        rating: Rating given by user to event
        history_u_lists: Dictionary of user's event history to update
        history_ur_lists: Dictionary of user's rating history to update
        history_v_lists: Dictionary of event's user history to update
        history_vr_lists: Dictionary of event's rating history to update

    Returns:
        Updated history_u_lists, history_ur_lists, history_v_lists, history_vr_lists
    """
    # Add event to user's history
    if user_id in history_u_lists:
        history_u_lists[user_id].append(event_id)
        history_ur_lists[user_id].append(rating)
    else:
        history_u_lists[user_id] = [event_id]
        history_ur_lists[user_id] = [rating]

    # Add user to event's history
    if event_id in history_v_lists:
        history_v_lists[event_id].append(user_id)
        history_vr_lists[event_id].append(rating)
    else:
        history_v_lists[event_id] = [user_id]
        history_vr_lists[event_id] = [rating]

    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists

