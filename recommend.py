import torch
from GraphRec_WWW19.graphrec_fixed import load_model_for_inference, add_new_user

################ GET RECOMMENDATIONS ####################

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

def get_recs(user_id):

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model for inference
    model, history_u_lists, history_v_lists, ratings_list = load_model_for_inference(
        'graphrec_model.pth', device)

    # Get all available events
    all_events = list(range(len(history_v_lists)))

    # Get top 10 recommendations
    recommendations = get_recommended_events(
        model, user_id, all_events, device, top_k=10, history_u_lists=history_u_lists)

    for i, (event_id, score) in enumerate(recommendations):
        print(f"{i + 1}. Event ID: {event_id}, Predicted Rating: {score:.2f}")

    return recommendations

# print(f"Top 10 recommendations for user 41:")
# get_recs(41)
