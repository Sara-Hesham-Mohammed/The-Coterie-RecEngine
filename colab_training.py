# Cell 1: Mount Google Drive
"""
# Run this cell first to mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
"""

# Cell 2: Install required packages
"""
# Install required packages
!pip install torch torchvision torchaudio
!pip install numpy pandas scikit-learn
!pip install requests
"""

# Cell 3: Clone repository and setup
"""
# Clone your repository (make sure it's public or you have access)
!git clone https://github.com/YOUR_USERNAME/The-Coterie-RecEngine.git
%cd The-Coterie-RecEngine
"""

# Cell 4: GPU Setup and Imports
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from GraphRec_WWW19.graphrec_fixed import GraphRec, train, test, save_model_with_metadata, memory_efficient_data_loading
from GraphRec_WWW19.UV_Encoders import UV_Encoder
from GraphRec_WWW19.UV_Aggregators import UV_Aggregator
from GraphRec_WWW19.Social_Encoders import Social_Encoder
from GraphRec_WWW19.Social_Aggregators import Social_Aggregator

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# If using GPU, print some additional info and set CUDA settings
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name()}")
    # Set default tensor type to cuda
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # Set seed for reproducibility
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Cell 5a: Set hyperparameters and load data
embed_dim = 64
epochs = 20
batch_size = 256
test_batch_size = 256
lr = 0.001
data_path = '/content/The-Coterie-RecEngine/Dataset/clean_meetup_data.pickle'

# Load data
print("Loading data...")
(history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
 train_loader, test_loader, social_adj_lists, ratings_list,
 num_users, num_items, num_ratings) = memory_efficient_data_loading(
    data_path=data_path,
    batch_size=batch_size,
    test_batch_size=test_batch_size,
    use_cuda=torch.cuda.is_available()
)

# Print detailed dataset statistics
print("\nDetailed Dataset Statistics:")
print(f"Number of users: {num_users}")
print(f"Number of items: {num_items}")
print(f"Number of ratings: {num_ratings}")
print(f"Training batches: {len(train_loader)}")
print(f"Testing batches: {len(test_loader)}")

# Check maximum indices
max_user_idx = max(
    max(idx for sublist in history_u_lists.values() for idx in sublist),
    max(idx for sublist in social_adj_lists.values() for idx in sublist),
    max(train_loader.dataset.tensors[0].max().item(),
        test_loader.dataset.tensors[0].max().item())
)
max_item_idx = max(
    max(idx for sublist in history_v_lists.values() for idx in sublist),
    max(train_loader.dataset.tensors[1].max().item(),
        test_loader.dataset.tensors[1].max().item())
)

print(f"\nMaximum indices in dataset:")
print(f"Max user index: {max_user_idx}")
print(f"Max item index: {max_item_idx}")

# Adjust num_users and num_items if necessary
num_users = max(num_users, max_user_idx + 1)
num_items = max(num_items, max_item_idx + 1)

print(f"\nAdjusted dimensions:")
print(f"Adjusted num_users: {num_users}")
print(f"Adjusted num_items: {num_items}")

# Recreate data loaders with proper device settings
if torch.cuda.is_available():
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    
    # Create samplers with CUDA generator
    train_sampler = torch.utils.data.RandomSampler(train_dataset, generator=torch.Generator(device='cuda'))
    test_sampler = torch.utils.data.RandomSampler(test_dataset, generator=torch.Generator(device='cuda'))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True
    )

# Cell 5b: Initialize model components
# Create embeddings with adjusted sizes
u2e = nn.Embedding(num_users, embed_dim).to(device)
v2e = nn.Embedding(num_items, embed_dim).to(device)
r2e = nn.Embedding(num_ratings, embed_dim).to(device)

# Ensure embeddings are on GPU if available
if torch.cuda.is_available():
    u2e = u2e.cuda()
    v2e = v2e.cuda()
    r2e = r2e.cuda()

# Create user-item aggregator and encoder
uv_aggregator = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device)
uv_encoder = UV_Encoder(
    features=v2e,
    embed_dim=embed_dim,
    history_uv_lists=history_v_lists,
    history_r_lists=history_vr_lists,
    aggregator=uv_aggregator,
    cuda=device
)

# Create a custom Social_Encoder class to handle device issues and index bounds
class CustomSocialEncoder(Social_Encoder):
    def forward(self, nodes):
        to_neighs = []
        for node in nodes:
            node_idx = int(node)
            # Get neighbors, handling the case where the node might not have any
            neighs = self.social_adj_lists.get(node_idx, [])
            # Ensure all neighbor indices are within bounds
            valid_neighs = [n for n in neighs if n < num_users]
            to_neighs.append(valid_neighs)

        # Get neighbor features through aggregator
        neigh_feats = self.aggregator.forward(nodes, to_neighs)

        # Keep everything on GPU
        if not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, device=self.device)
        elif nodes.device != self.device:
            nodes = nodes.to(self.device)

        # Ensure nodes are within bounds
        nodes = torch.clamp(nodes, 0, num_users - 1)

        # Get self features
        self_feats = self.features(nodes)
        
        # Make sure dimensions match
        if self_feats.dim() == 2:
            if self_feats.size(0) != neigh_feats.size(0):
                self_feats = self_feats.t()
        
        # Concatenate and apply linear transformation
        combined = torch.cat([self_feats, neigh_feats], dim=1)
        combined = F.relu(self.linear1(combined))
        
        return combined

# Create social aggregator and encoder with modified features function
def features_to_device(nodes):
    if not isinstance(nodes, torch.Tensor):
        nodes = torch.tensor(nodes, device=device)
    elif nodes.device != device:
        nodes = nodes.to(device)
    
    # Ensure nodes are within bounds
    nodes = torch.clamp(nodes, 0, num_users - 1)
    
    feats = u2e(nodes)
    # Ensure correct shape
    if feats.dim() == 2 and feats.size(0) != nodes.size(0):
        feats = feats.t()
    return feats

social_aggregator = Social_Aggregator(features_to_device, u2e, embed_dim, cuda=device)
social_encoder = CustomSocialEncoder(
    features=u2e,
    embed_dim=embed_dim,
    social_adj_lists=social_adj_lists,
    aggregator=social_aggregator,
    base_model=None,
    cuda=device
)

# Move encoders to device
uv_encoder = uv_encoder.to(device)
social_encoder = social_encoder.to(device)

# Initialize the model
model = GraphRec(social_encoder, uv_encoder, r2e).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Cell 5c: Training loop
best_rmse = float('inf')
best_mae = float('inf')
model_save_path = '/content/drive/MyDrive/coterie_model.pth'

for epoch in range(1, epochs + 1):
    loss = train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
    rmse, mae = test(model, device, test_loader)
    
    # Save if best model
    if rmse < best_rmse:
        best_rmse = rmse
        best_mae = mae
        save_model_with_metadata(
            model=model,
            model_path=model_save_path,
            data_path=data_path,
            embed_dim=embed_dim,
            epoch=epoch,
            optimizer_state=optimizer.state_dict()
        )
    
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}') 