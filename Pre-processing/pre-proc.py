import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
import torch
import time

print("Starting data processing with GPU acceleration...")
start_time = time.time()

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("CUDA not available, falling back to CPU")
    device = torch.device("cpu")

# Load your CSV files
print("Loading CSV files...")


# Set a consistent random seed for reproducibility
RANDOM_SEED = 42
FRAC = 0.025 # 2.5% sample

print("Sampling 2.5% of each dataset...")

user_event_0 = pd.read_csv('../Dataset/cleaned_user_event_0.csv').sample(frac=FRAC, random_state=RANDOM_SEED)
user_event_1 = pd.read_csv('../Dataset/cleaned_user_event_1.csv').sample(frac=FRAC, random_state=RANDOM_SEED)
user_group = pd.read_csv('../Dataset/cleaned_user_group.csv').sample(frac=FRAC, random_state=RANDOM_SEED)
user_tag = pd.read_csv('../Dataset/cleaned_user_tag.csv').sample(frac=FRAC, random_state=RANDOM_SEED)
event_group = pd.read_csv('../Dataset/cleaned_event_group.csv').sample(frac=FRAC, random_state=RANDOM_SEED)
group_tag = pd.read_csv('../Dataset/cleaned_group_tag.csv').sample(frac=FRAC, random_state=RANDOM_SEED)
tag_text = pd.read_csv('../Dataset/cleaned_tag_text.csv').sample(frac=FRAC, random_state=RANDOM_SEED)

print("Sampling completed.")


# Combine user-event interactions
print("Combining user-event data...")
user_event = pd.concat([user_event_0, user_event_1])
# Assign synthetic ratings from a realistic distribution (weighted towards higher ratings)
rating_choices = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
rating_probs =   [0.05, 0.05, 0.1, 0.15, 0.2, 0.15, 0.15, 0.1, 0.05]  # sum to 1

np.random.seed(RANDOM_SEED)
user_event['rating'] = np.random.choice(rating_choices, size=len(user_event), p=rating_probs)


# Create unique IDs for users and events
print("Creating user and event mappings...")
unique_users = user_event['UserID'].unique()
unique_events = user_event['EventID'].unique()

# Re-indexing - use dictionaries for faster mapping
user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
event_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_events)}

# Map to new IDs - use vectorized operations
user_event['user_id_new'] = user_event['UserID'].map(user_id_map)
user_event['event_id_new'] = user_event['EventID'].map(event_id_map)

# Create social connections from user_group
print("Building social connections...")
social_adj_lists = defaultdict(set)

# Group users by group ID directly - much more efficient
grouped_dict = user_group.groupby('groupID')['userID'].apply(list).to_dict()

# Process in parallel using GPU where beneficial
for group_id, users in grouped_dict.items():
    # Filter users that are in our mapping
    valid_users = [user_id_map.get(user, None) for user in users]
    valid_users = [u for u in valid_users if u is not None]

    # Skip tiny groups
    if len(valid_users) <= 1:
        continue

    # Create connections directly
    for i, user1 in enumerate(valid_users):
        # Add all other users at once - much faster than looping
        others = set(valid_users) - {user1}
        social_adj_lists[user1].update(others)

print(f"Created social connections for {len(social_adj_lists)} users")


# Create history lists efficiently
print("Building history lists...")
history_u_lists = defaultdict(list)  # user's event history
history_ur_lists = defaultdict(list)  # user's rating history
history_v_lists = defaultdict(list)  # event's user history
history_vr_lists = defaultdict(list)  # event's rating history

# Group by user_id_new to efficiently build user history
user_grouped = user_event.groupby('user_id_new')
for user_id, group in user_grouped:
    history_u_lists[user_id] = group['event_id_new'].tolist()
    history_ur_lists[user_id] = group['rating'].tolist()

# Group by event_id_new to efficiently build event history
event_grouped = user_event.groupby('event_id_new')
for event_id, group in event_grouped:
    history_v_lists[event_id] = group['user_id_new'].tolist()
    history_vr_lists[event_id] = group['rating'].tolist()

print(f"Created history lists for {len(history_u_lists)} users and {len(history_v_lists)} events")


# Split into train and test sets (80-20 split)
print("Creating train/test split...")
np.random.seed(1234)

# Use GPU for random selection
indices = torch.randperm(len(user_event), device=device)
split_idx = int(0.8 * len(user_event))
train_indices = indices[:split_idx].cpu().numpy()
test_indices = indices[split_idx:].cpu().numpy()

train_data = user_event.iloc[train_indices]
test_data = user_event.iloc[test_indices]

train_u = train_data['user_id_new'].tolist()
train_v = train_data['event_id_new'].tolist()
train_r = train_data['rating'].tolist()

test_u = test_data['user_id_new'].tolist()
test_v = test_data['event_id_new'].tolist()
test_r = test_data['rating'].tolist()

# Define rating scale
ratings_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Prepare the data in the required format
data = (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
        train_u, train_v, train_r, test_u, test_v, test_r,
        social_adj_lists, ratings_list)

# Save as pickle file
print("Saving data to pickle file...")
with open('clean_meetup_data.pickle', 'wb') as f:
    pickle.dump(data, f)

end_time = time.time()
print(f"Data processing completed in {end_time - start_time:.2f} seconds")