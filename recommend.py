import torch
from GraphRec_WWW19.graphrec_fixed import load_model_for_inference, add_new_user, get_recommended_events

################ GET RECOMMENDATIONS ####################

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
