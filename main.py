import argparse
import os
import pickle
import torch
import torch.nn as nn

from GraphRec_WWW19.Social_Aggregators import Social_Aggregator
from GraphRec_WWW19.Social_Encoders import Social_Encoder
from GraphRec_WWW19.UV_Aggregators import UV_Aggregator
from GraphRec_WWW19.UV_Encoders import UV_Encoder
from GraphRec_WWW19.graphrec_fixed import GraphRec, save_model_with_metadata, load_model_for_inference, test, \
    get_recommended_events, train

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
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'recommend'],
                        help='train, test or recommend')
    parser.add_argument('--user_id', type=int, default=None, help='user ID for recommendations')
    parser.add_argument('--top_k', type=int, default=10, help='number of recommendations')
    args = parser.parse_args()

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    embed_dim = args.embed_dim

    if args.mode == 'train' or args.mode == 'test':
        # Load data
        print(f"Loading data from {args.data_path}")
        with open(args.data_path, 'rb') as data_file:
            history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, social_adj_lists, ratings_list = pickle.load(
                data_file)

        trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                                  torch.FloatTensor(train_r))
        testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                                 torch.FloatTensor(test_r))
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
        num_users = history_u_lists.__len__()
        num_items = history_v_lists.__len__()
        num_ratings = ratings_list.__len__()

        print(f"Data loaded with {num_users} users and {num_items} events")

        u2e = nn.Embedding(num_users, embed_dim).to(device)
        v2e = nn.Embedding(num_items, embed_dim).to(device)
        r2e = nn.Embedding(num_ratings, embed_dim).to(device)

        # user feature
        # features: item * rating
        agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
        enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device,
                                   uv=True)
        # neighobrs
        agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
        enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
                               base_model=enc_u_history, cuda=device)

        # item feature: user * rating
        agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
        enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device,
                                   uv=False)

        # model
        graphrec = GraphRec(enc_u, enc_v_history, r2e).to(device)
        optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=args.lr, alpha=0.9)

        if args.mode == 'train':
            best_rmse = 9999.0
            best_mae = 9999.0
            endure_count = 0

            for epoch in range(1, args.epochs + 1):
                train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
                expected_rmse, mae = test(graphrec, device, test_loader)

                # early stopping
                if best_rmse > expected_rmse:
                    best_rmse = expected_rmse
                    best_mae = mae
                    endure_count = 0
                    # Save the best model
                    save_model_with_metadata(graphrec, args.model_path, args.data_path, embed_dim)
                else:
                    endure_count += 1
                print(f"Epoch {epoch}: RMSE: {expected_rmse:.4f}, MAE: {mae:.4f}")

                if endure_count > 5:
                    print("Early stopping!")
                    break

            print(f"Training complete. Best RMSE: {best_rmse:.4f}, Best MAE: {best_mae:.4f}")

        elif args.mode == 'test':
            # Load the trained model
            print(f"Loading model from {args.model_path}")
            metadata = torch.load(args.model_path, map_location=device)
            graphrec.load_state_dict(metadata['model_state'])

            # Test the model
            expected_rmse, mae = test(graphrec, device, test_loader)
            print(f"Test results - RMSE: {expected_rmse:.4f}, MAE: {mae:.4f}")

    elif args.mode == 'recommend':
        if args.user_id is None:
            print("Error: User ID required for recommendations")
            return

        # Load model
        model, history_u_lists, history_v_lists, ratings_list = load_model_for_inference(
            args.model_path, device)

        # Get all event IDs
        all_events = list(range(len(history_v_lists)))

        # Get recommendations
        recommendations = get_recommended_events(
            model, args.user_id, all_events, device,
            top_k=args.top_k, history_u_lists=history_u_lists)

        print(f"Top {args.top_k} recommendations for user {args.user_id}:")
        for i, (event_id, score) in enumerate(recommendations):
            print(f"{i + 1}. Event ID: {event_id}, Predicted Rating: {score:.2f}")


if __name__ == "__main__":
    main()