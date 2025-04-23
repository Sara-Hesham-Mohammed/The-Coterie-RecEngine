import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

# Load your CSV files
user_event_0 = pd.read_csv('../Dataset/cleaned_user_event_0.csv')
user_event_1 = pd.read_csv('../Dataset/cleaned_user_event_1.csv')
user_group = pd.read_csv('../Dataset/cleaned_user_group.csv')
user_tag = pd.read_csv('../Dataset/cleaned_user_tag.csv')
event_group = pd.read_csv('../Dataset/cleaned_event_group.csv')
group_tag = pd.read_csv('../Dataset/cleaned_group_tag.csv')
tag_text = pd.read_csv('../Dataset/cleaned_tag_text.csv')

# Combine user-event interactions
user_event = pd.concat([user_event_0, user_event_1])

# Since you don't have ratings, create a default rating (e.g., 1.0 for attended)
# Or use a scale like 2.5 as you suggested
user_event['rating'] = 2.5

# Create unique IDs for users and events
unique_users = user_event['UserID'].unique()
unique_events = user_event['EventID'].unique()

## Re indexing ##
user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
event_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_events)}

# Map to new IDs
user_event['user_id_new'] = user_event['UserID'].map(user_id_map)
user_event['event_id_new'] = user_event['EventID'].map(event_id_map)

# Create social connections from user_group
# Assume users in the same group have a connection
social_adj_lists = defaultdict(set)

# Process each group to create connections
for group_id in user_group['groupID'].unique():
    users_in_group = user_group[user_group['groupID'] == group_id]['userID'].tolist()
    users_in_group = [user_id_map.get(user, user) for user in users_in_group if user in user_id_map]

    # Create connections between all users in the same group
    for user in users_in_group:
        social_adj_lists[user].update([u for u in users_in_group if u != user])

# Create history lists
history_u_lists = defaultdict(list)  # user's event history
history_ur_lists = defaultdict(list)  # user's rating history
history_v_lists = defaultdict(list)  # event's user history
history_vr_lists = defaultdict(list)  # event's rating history

for _, row in user_event.iterrows():
    u = row['user_id_new']
    v = row['event_id_new']
    r = row['rating']

    history_u_lists[u].append(v)
    history_ur_lists[u].append(r)
    history_v_lists[v].append(u)
    history_vr_lists[v].append(r)

# Split into train and test sets (80-20 split)
np.random.seed(1234)
train_indices = np.random.choice(len(user_event), int(0.8 * len(user_event)), replace=False)
test_indices = np.array(list(set(range(len(user_event))) - set(train_indices)))

train_data = user_event.iloc[train_indices]
test_data = user_event.iloc[test_indices]

train_u = train_data['user_id_new'].tolist()
train_v = train_data['event_id_new'].tolist()
train_r = train_data['rating'].tolist()

test_u = test_data['user_id_new'].tolist()
test_v = test_data['event_id_new'].tolist()
test_r = test_data['rating'].tolist()

# Define rating scale (you might need to adjust this)
ratings_list = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Prepare the data in the required format
data = (history_u_lists, history_ur_lists, history_v_lists, history_vr_lists,
        train_u, train_v, train_r, test_u, test_v, test_r,
        social_adj_lists, ratings_list)

# Save as pickle file
with open('clean_meetup_data.pickle', 'wb') as f:
    pickle.dump(data, f)