
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

            print(f"Training complete. Best RMSE: {best_rmse:.4f}, Best MAE: {best_mae:.4f}")

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