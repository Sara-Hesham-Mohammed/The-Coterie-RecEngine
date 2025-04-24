import torch
"""
    Generates recommendations for a user using the trained GraphRec model.

    Args:
        model: Trained GraphRec model
        user_id: The internal ID of the user to recommend for
        all_events: List of all event IDs to consider (that are gonna be pulled using the API)
        device: Device to run inference on (CPU or CUDA)
        top_k: Number of recommendations to return
        history_u_lists: list of events user has interacted with before (perhaps collect them using details like attended/favorited/atc.)
    Returns:
        List of (event_id, predicted_rating) tuples
"""
def recommend_events(model, user_id, all_events, device, top_k=10, history_u_lists=None):

    model.eval()

    # Get events the user hasn't interacted with
    if user_id in history_u_lists:
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
            test_u = torch.full((len(batch_events),), user_id, dtype=torch.long).to(device)
            test_v = torch.tensor(batch_events, dtype=torch.long).to(device)

            # Get predictions
            pred_ratings = model.forward(test_u, test_v)

            # Store results
            for j, event_id in enumerate(batch_events):
                predictions.append((event_id, pred_ratings[j].item()))

    # Sort by predicted rating
    sorted_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    # Return top-k recommendations
    return sorted_predictions[:top_k]