import gc
import os
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
import argparse

"""
GraphRec: Graph Neural Networks for Social Recommendation. 
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin. 
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

Fixed implementation to address RMSE/MAE not changing issue.
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

        # Initialize weights with Xavier/Glorot initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, nodes_u, nodes_v):
        """
        Forward pass without silent error handling that masks issues
        """
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        # Check for NaN values early
        if torch.isnan(embeds_u).any() or torch.isnan(embeds_v).any():
            print("Warning: NaN values detected in embeddings")
            # It's better to raise an exception than to silently continue
            raise ValueError("NaN values in embeddings")

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training, p=0.3)  # Explicit dropout probability
        x_u = self.w_ur2(x_u)

        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training, p=0.3)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training, p=0.3)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training, p=0.3)
        scores = self.w_uv3(x)

        # Clamp scores to a reasonable rating range (e.g., 1-5)
        scores = torch.clamp(scores, min=1.0, max=5.0)

        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0

    # Track batch loss values to identify problematic batches
    batch_losses = []

    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        # Move data to device
        batch_nodes_u, batch_nodes_v, labels_list = batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(
            device)

        # Print stats about the current batch
        if i % 100 == 0:
            print(
                f"Batch {i} - Users: {batch_nodes_u.shape}, Items: {batch_nodes_v.shape}, Labels: {labels_list.shape}")
            print(f"Labels range: {labels_list.min().item():.2f} to {labels_list.max().item():.2f}")

        # Zero gradients for every batch
        optimizer.zero_grad()

        try:
            # Compute loss
            loss = model.loss(batch_nodes_u, batch_nodes_v, labels_list)

            # Check if loss is valid
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: Invalid loss value {loss.item()} in batch {i}")
                continue

            # Backpropagation
            loss.backward()  # Removed retain_graph=True which can cause memory leaks

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            # Record loss
            batch_loss = loss.item()
            running_loss += batch_loss
            batch_losses.append(batch_loss)

            if i % 100 == 0:
                print(
                    f'[{epoch}, {i}] loss: {running_loss / 100:.3f}, The best rmse/mae: {best_rmse:.6f} / {best_mae:.6f}')
                running_loss = 0.0

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Skip this batch but continue training
            continue

    # Analyze batch losses
    if batch_losses:
        print(
            f"Epoch {epoch} batch loss stats - Min: {min(batch_losses):.4f}, Max: {max(batch_losses):.4f}, Avg: {sum(batch_losses) / len(batch_losses):.4f}")

    return np.mean(batch_losses) if batch_losses else float('inf')


def test(model, device, test_loader):
    model.eval()
    predictions = []
    targets = []

    # Count successful and failed batches
    success_count = 0
    fail_count = 0

    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)

            try:
                # Get predictions
                val_output = model.forward(test_u, test_v)

                # Check predictions for validity
                if torch.isnan(val_output).any():
                    print(f"Warning: NaN in predictions, skipping batch")
                    fail_count += 1
                    continue

                # Store predictions and targets
                batch_predictions = val_output.detach().cpu().numpy()
                batch_targets = tmp_target.detach().cpu().numpy()

                predictions.extend(batch_predictions)
                targets.extend(batch_targets)

                success_count += 1

                # Print validation statistics periodically
                if success_count % 50 == 0:
                    print(f"Processed {success_count} test batches successfully")
                    if predictions:
                        print(f"Current predictions range: {min(predictions):.4f} to {max(predictions):.4f}")

            except Exception as e:
                print(f"Error in test batch: {e}")
                fail_count += 1

    # Convert to numpy arrays for metric calculation
    predictions = np.array(predictions)
    targets = np.array(targets)

    # Check if we have valid predictions
    if len(predictions) == 0:
        print("Error: No valid predictions generated during testing")
        return 9999.0, 9999.0

    # Log information about predictions
    print(f"Test completed - Successful batches: {success_count}, Failed batches: {fail_count}")
    print(f"Predictions shape: {predictions.shape}, Targets shape: {targets.shape}")
    print(f"Predictions range: {np.min(predictions):.4f} to {np.max(predictions):.4f}")
    print(f"Targets range: {np.min(targets):.4f} to {np.max(targets):.4f}")

    # Detailed analysis of prediction distribution
    print("Prediction distribution:")
    hist, bins = np.histogram(predictions, bins=10)
    for i in range(len(hist)):
        print(f"  {bins[i]:.2f}-{bins[i + 1]:.2f}: {hist[i]} items ({hist[i] / len(predictions) * 100:.1f}%)")

    try:
        # Calculate metrics
        rmse = sqrt(mean_squared_error(predictions, targets))
        mae = mean_absolute_error(predictions, targets)
        print(f"Calculated RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return rmse, mae
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return 9999.0, 9999.0


def save_model_with_metadata(model, model_path, data_path, embed_dim, epoch=None, optimizer_state=None):
    """
    Enhanced save model with additional metadata for training continuation
    """
    metadata = {
        'model_state': model.state_dict(),
        'data_path': data_path,
        'embed_dim': embed_dim,
        'epoch': epoch,
        'optimizer_state': optimizer_state if optimizer_state else None
    }
    torch.save(metadata, model_path)
    print(f"Model saved to {model_path} with metadata (epoch {epoch})")


def normalize_data(ratings, min_val=1.0, max_val=5.0):
    """
    Normalize ratings to the range [0, 1]
    """
    normalized = (ratings - min_val) / (max_val - min_val)
    return normalized


def denormalize_predictions(normalized_preds, min_val=1.0, max_val=5.0):
    """
    Denormalize predictions back to the original rating scale
    """
    denormalized = normalized_preds * (max_val - min_val) + min_val
    return denormalized


def memory_efficient_data_loading(data_path, batch_size, test_batch_size, use_cuda=True, normalize=True):
    """
    Memory-efficient data loading for GraphRec with normalization option
    """
    print(f"Loading data from {data_path}")

    # Force garbage collection before loading data
    gc.collect()

    # Load data incrementally
    try:
        with open(data_path, 'rb') as data_file:
            data = pickle.load(data_file)
            print("Pickle loaded, unpacking data...")

            history_u_lists = data[0]
            history_ur_lists = data[1]
            history_v_lists = data[2]
            history_vr_lists = data[3]
            train_u = data[4]
            train_v = data[5]
            train_r = data[6]
            test_u = data[7]
            test_v = data[8]
            test_r = data[9]
            social_adj_lists = data[10]
            ratings_list = data[11]

            # Delete the combined data structure to free memory
            del data
            gc.collect()
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # Check device setup
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    pin_memory = use_cuda and torch.cuda.is_available()

    # Process data as numpy arrays
    train_u = np.array(train_u, dtype=np.int64)
    train_v = np.array(train_v, dtype=np.int64)
    train_r = np.array(train_r, dtype=np.float32)
    test_u = np.array(test_u, dtype=np.int64)
    test_v = np.array(test_v, dtype=np.int64)
    test_r = np.array(test_r, dtype=np.float32)

    # Data analysis: Check distribution of ratings
    print("\nData Statistics:")
    print(
        f"Train ratings - min: {np.min(train_r):.2f}, max: {np.max(train_r):.2f}, mean: {np.mean(train_r):.2f}, std: {np.std(train_r):.2f}")
    print(
        f"Test ratings - min: {np.min(test_r):.2f}, max: {np.max(test_r):.2f}, mean: {np.mean(test_r):.2f}, std: {np.std(test_r):.2f}")

    # Calculate rating distribution
    unique_ratings = np.unique(np.concatenate([train_r, test_r]))
    train_hist = np.histogram(train_r, bins=len(unique_ratings))[0]
    test_hist = np.histogram(test_r, bins=len(unique_ratings))[0]

    print("\nRating Distribution:")
    for i, rating in enumerate(unique_ratings):
        train_percent = train_hist[i] / len(train_r) * 100 if i < len(train_hist) else 0
        test_percent = test_hist[i] / len(test_r) * 100 if i < len(test_hist) else 0
        print(f"Rating {rating:.1f}: Train {train_percent:.1f}%, Test {test_percent:.1f}%")

    # Normalize ratings if requested
    if normalize:
        print("\nNormalizing ratings to [0, 1] range...")
        min_rating = min(np.min(train_r), np.min(test_r))
        max_rating = max(np.max(train_r), np.max(test_r))

        train_r = normalize_data(train_r, min_rating, max_rating)
        test_r = normalize_data(test_r, min_rating, max_rating)

        print(f"Normalized train ratings - min: {np.min(train_r):.2f}, max: {np.max(train_r):.2f}")
        print(f"Normalized test ratings - min: {np.min(test_r):.2f}, max: {np.max(test_r):.2f}")

    # Create TensorDatasets
    trainset = torch.utils.data.TensorDataset(
        torch.from_numpy(train_u),
        torch.from_numpy(train_v),
        torch.from_numpy(train_r)
    )

    testset = torch.utils.data.TensorDataset(
        torch.from_numpy(test_u),
        torch.from_numpy(test_v),
        torch.from_numpy(test_r)
    )

    # Clear numpy arrays to free memory
    del train_u, train_v, train_r, test_u, test_v, test_r
    gc.collect()

    # Create data loaders with worker settings for memory efficiency
    num_workers = 0  # Start with 0 and increase if memory allows
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    num_users = len(history_u_lists)
    num_items = len(history_v_lists)
    num_ratings = len(ratings_list)

    print(f"Data loaded with {num_users} users and {num_items} events")

    return (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
            train_loader, test_loader, social_adj_lists, ratings_list,
            num_users, num_items, num_ratings)


def check_gradient_flow(model, epoch):
    """
    Check gradient flow through the model to detect potential issues
    """
    print(f"\nEpoch {epoch} - Checking gradient flow:")
    ave_grads = []
    max_grads = []
    layers = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().cpu().item())
            max_grads.append(param.grad.abs().max().cpu().item())

            # Print out extreme gradient values
            if param.grad.abs().max().cpu().item() > 10:
                print(f"  Warning: High gradient in {name}: {param.grad.abs().max().cpu().item():.4f}")
            if param.grad.abs().mean().cpu().item() < 1e-5:
                print(f"  Warning: Low gradient in {name}: {param.grad.abs().mean().cpu().item():.8f}")

    # Print overall stats
    if ave_grads:
        print(f"  Average gradient: {sum(ave_grads) / len(ave_grads):.6f}")
        print(f"  Max gradient: {max(max_grads):.6f}")
    else:
        print("  No gradients available")



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphRec model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    parser.add_argument('--data_path', type=str, default='clean_meetup_data.pickle', help='path to data pickle file')
    parser.add_argument('--model_path', type=str, default='graphrec_model.pth', help='path to save/load model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='train or test')
    parser.add_argument('--normalize', action='store_true', help='normalize ratings to [0, 1]')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay for regularization')
    parser.add_argument('--scheduler', action='store_true', help='use learning rate scheduler')
    parser.add_argument('--resume', action='store_true', help='resume training from saved model')
    args = parser.parse_args()

    # Set device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    embed_dim = args.embed_dim

    try:
        # Load data with normalization option
        print(f"Loading data from {args.data_path}")
        (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
         train_loader, test_loader, social_adj_lists, ratings_list,
         num_users, num_items, num_ratings) = memory_efficient_data_loading(
            args.data_path, args.batch_size, args.test_batch_size, use_cuda, args.normalize)

        # Initialize embeddings with Xavier/Glorot initialization
        u2e = nn.Embedding(num_users, embed_dim).to(device)
        v2e = nn.Embedding(num_items, embed_dim).to(device)
        r2e = nn.Embedding(num_ratings, embed_dim).to(device)

        # Initialize embeddings properly
        nn.init.xavier_uniform_(u2e.weight)
        nn.init.xavier_uniform_(v2e.weight)
        nn.init.xavier_uniform_(r2e.weight)

        # Create aggregators and encoders
        agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
        enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device,
                                   uv=True)

        agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
        enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                               base_model=enc_u_history, cuda=device)

        agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
        enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device,
                                   uv=False)

        # Build GraphRec model
        graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)

        # Use Adam optimizer with weight decay for regularization
        optimizer = torch.optim.Adam(graphrec.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Learning rate scheduler
        scheduler = None
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )

        start_epoch = 0
        best_rmse = 9999.0
        best_mae = 9999.0

        # Resume training if requested
        if args.resume and os.path.exists(args.model_path):
            print(f"Resuming training from {args.model_path}")
            metadata = torch.load(args.model_path, map_location=device)
            graphrec.load_state_dict(metadata['model_state'])

            if metadata.get('optimizer_state'):
                optimizer.load_state_dict(metadata['optimizer_state'])

            if metadata.get('epoch'):
                start_epoch = metadata['epoch'] + 1
                print(f"Resuming from epoch {start_epoch}")

        if args.mode == 'train':
            endure_count = 0

            # Initial evaluation
            print("Initial evaluation:")
            initial_rmse, initial_mae = test(graphrec, device, test_loader)
            print(f"Initial RMSE: {initial_rmse:.4f}, MAE: {initial_mae:.4f}")

            for epoch in range(start_epoch, args.epochs + 1):
                print(f"\n===== Epoch {epoch} =====")

                # Train the model
                avg_loss = train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)

                # Check gradient flow every few epochs
                if epoch % 5 == 0:
                    check_gradient_flow(graphrec, epoch)

                # Test the model
                current_rmse, current_mae = test(graphrec, device, test_loader)
                print(
                    f"Epoch {epoch} evaluation - RMSE: {current_rmse:.4f}, MAE: {current_mae:.4f}, Loss: {avg_loss:.4f}")

                # Update learning rate if using scheduler
                if scheduler:
                    scheduler.step(current_rmse)
                    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

                # Save model if improved
                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_mae = current_mae
                    endure_count = 0
                    save_model_with_metadata(graphrec, args.model_path, args.data_path, embed_dim,
                                             epoch, optimizer.state_dict())
                    print(f"New best model saved with RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}")
                else:
                    endure_count += 1
                    print(f"No improvement for {endure_count} epochs")

                # Save checkpoint every 10 epochs regardless of performance
                if epoch % 10 == 0:
                    checkpoint_path = f"{args.model_path}.ep{epoch}"
                    save_model_with_metadata(graphrec, checkpoint_path, args.data_path, embed_dim,
                                             epoch, optimizer.state_dict())
                    print(f"Checkpoint saved to {checkpoint_path}")

                # Early stopping
                if endure_count > 5:
                    print("Early stopping triggered!")
                    break





            print(f"TRAINING COMPLETE. Best RMSE: {best_rmse:.4f}, Best MAE: {best_mae:.4f}")
            print("\nTraining complete. Model saved to:", args.model_path)

        elif args.mode == 'test':
            # Load the trained model
            print(f"Loading model from {args.model_path}")
            metadata = torch.load(args.model_path, map_location=device)
            graphrec.load_state_dict(metadata['model_state'])
            graphrec.eval()

            # Test the model
            rmse, mae = test(graphrec, device, test_loader)
            print(f"Test results - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
