import torch
from GraphRec_WWW19.graphrec_fixed import load_model_for_inference, add_new_user, get_recommended_events

# TODO: DONT RUN THIS YET. UNFINISHED
############### ADD NEW USER #############

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists = load_model_for_inference(
    'graphrec_model.pth', device, return_all_data=True)

# Get next available user ID
new_user_id = len(history_u_lists)
print(f"Creating new user with ID: {new_user_id}")

# Define social connections for the new user (group memberships)
# This is crucial for cold-start recommendation
social_connections = [0, 5, 10]  # Example: connect to users 0, 5, and 10
print(f"Adding social connections: {social_connections}")

# Add the new user
history_u_lists, history_ur_lists, social_adj_lists = add_new_user(
    new_user_id, social_connections, history_u_lists, history_ur_lists, social_adj_lists)

# Get recommendations for the new user
all_events = list(range(len(history_v_lists)))
recommendations = get_recommended_events(
    model, new_user_id, all_events, device, top_k=10, history_u_lists=history_u_lists)

print(f"Top 10 recommendations for new user {new_user_id}:")
