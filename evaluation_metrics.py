import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_rating_model(model, test_data):
    model.eval()
    with torch.no_grad():
        preds = model(test_data.x, test_data.edge_index).squeeze()
        y_true = test_data.y
        rmse = mean_squared_error(y_true, preds, squared=False)
        mae = mean_absolute_error(y_true, preds)
        r2 = r2_score(y_true, preds)
    return {"RMSE": rmse, "MAE": mae, "R²": r2}




def evaluate_topk(model, test_data, k=10):
    model.eval()
    with torch.no_grad():
        scores = model(test_data.x, test_data.edge_index)  # Shape: [users, items]
        topk_items = scores.topk(k, dim=1).indices  # Top-K item indices per user

        # Assume `test_data.relevant_items` is a list of ground-truth liked items per user
        hits = []
        for user, items in enumerate(topk_items):
            hits.append(len(set(items.tolist()) & set(test_data.relevant_items[user])))

        precision_at_k = np.mean([hits / k for hits in hits])
        recall_at_k = np.mean([hits / len(test_data.relevant_items[u]) for u, hits in enumerate(hits)])

        return {"Precision@K": precision_at_k, "Recall@K": recall_at_k}