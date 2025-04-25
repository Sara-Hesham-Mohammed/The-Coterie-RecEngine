from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


# Helper function for constraints
def is_valid(user, cluster):
    if not cluster:  # If cluster is empty, it's valid
        return True

    age_diffs = [abs(user["age"] - other["age"]) for other in cluster]
    if any(diff < 10 for diff in age_diffs):  # Constraint 1: too close in age
        return False

    for other in cluster:
        sim = jaccard(user["tags"], other["tags"])
        if sim > 0.7 or sim < 0.2:  # Constraint 2: tag similarity constraints
            return False
    return True


def jaccard(set1, set2):
    return len(set1 and set2) / len(set1 or set2)


def get_clusters(users, initial_clusters=3):
    # Convert tags to vectors for initial clustering
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform([user["tags"] for user in users])

    # Add age as a feature (normalized)
    ages = np.array([user["age"] for user in users]).reshape(-1, 1)
    ages_normalized = (ages - ages.mean()) / ages.std()

    # Combine features
    features = np.hstack([tag_matrix, ages_normalized])

    # Get initial cluster suggestions using KMeans
    kmeans = KMeans(n_clusters=initial_clusters)
    suggested_labels = kmeans.fit_predict(features)

    # Create initial suggested clusters
    suggested_clusters = [[] for _ in range(initial_clusters)]
    for user, label in zip(users, suggested_labels):
        suggested_clusters[label].append(user)

    # Final clusters that respect constraints
    final_clusters = []

    # Process each suggested cluster
    for suggested_cluster in suggested_clusters:
        valid_sub_cluster = []

        # Try to keep users together from the suggested cluster if they're valid
        for user in suggested_cluster:
            if is_valid(user, valid_sub_cluster):
                valid_sub_cluster.append(user)
            else:
                # Try adding to existing final clusters
                added = False
                for cluster in final_clusters:
                    if is_valid(user, cluster):
                        cluster.append(user)
                        added = True
                        break

                # Create new cluster if needed
                if not added:
                    final_clusters.append([user])

        # Add the valid sub-cluster if it's not empty
        if valid_sub_cluster:
            final_clusters.append(valid_sub_cluster)

    # Process any remaining users who didn't fit well
    processed_users = sum([len(cluster) for cluster in final_clusters])
    if processed_users < len(users):
        # Create a new cluster for remaining users
        remaining_cluster = []
        for user in users:
            # Check if user is already in a cluster
            found = False
            for cluster in final_clusters:
                if any(u["id"] == user["id"] for u in cluster):
                    found = True
                    break

            # If not found, add to remaining or create new cluster
            if not found:
                if is_valid(user, remaining_cluster):
                    remaining_cluster.append(user)
                else:
                    final_clusters.append([user])

        # Add remaining cluster if not empty
        if remaining_cluster:
            final_clusters.append(remaining_cluster)

    return final_clusters

# Example usage
if __name__ == "__main__":
    users = [
    {"id": 1, "age": 25, "tags": ["sports", "tech", "music", "gaming", "travel"]},
    {"id": 2, "age": 40, "tags": ["cooking", "music", "tech", "finance", "reading"]},
    {"id": 3, "age": 33, "tags": ["travel", "photography", "tech", "fitness", "music"]},
    {"id": 4, "age": 22, "tags": ["gaming", "tech", "anime", "music", "sports"]},
    {"id": 5, "age": 55, "tags": ["gardening", "cooking", "reading", "history", "travel"]},
    {"id": 6, "age": 29, "tags": ["music", "travel", "sports", "fitness", "food"]},
    {"id": 7, "age": 61, "tags": ["history", "reading", "gardening", "cooking", "finance"]},
    {"id": 8, "age": 45, "tags": ["finance", "tech", "sports", "news", "reading"]},
    {"id": 9, "age": 37, "tags": ["fitness", "cooking", "music", "photography", "health"]},
    {"id": 10, "age": 19, "tags": ["anime", "gaming", "sports", "tech", "memes"]},
    {"id": 11, "age": 48, "tags": ["travel", "history", "finance", "tech", "reading"]},
    {"id": 12, "age": 31, "tags": ["tech", "music", "gaming", "food", "anime"]}
]

    clusters = get_clusters(users)
    print(clusters)