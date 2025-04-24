import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import random


# Sample user data, fetch it tho using the api and User class u made
users = [
    {"id": 1, "age": 25, "tags": {"sports", "tech", "music", "gaming", "travel"}},
    {"id": 2, "age": 40, "tags": {"cooking", "music", "tech", "finance", "reading"}},
    {"id": 3, "age": 33, "tags": {"travel", "photography", "tech", "fitness", "music"}},
    {"id": 4, "age": 22, "tags": {"gaming", "tech", "anime", "music", "sports"}},
    {"id": 5, "age": 55, "tags": {"gardening", "cooking", "reading", "history", "travel"}},
    {"id": 6, "age": 29, "tags": {"music", "travel", "sports", "fitness", "food"}},
    {"id": 7, "age": 61, "tags": {"history", "reading", "gardening", "cooking", "finance"}},
    {"id": 8, "age": 45, "tags": {"finance", "tech", "sports", "news", "reading"}},
    {"id": 9, "age": 37, "tags": {"fitness", "cooking", "music", "photography", "health"}},
    {"id": 10, "age": 19, "tags": {"anime", "gaming", "sports", "tech", "memes"}},
    {"id": 11, "age": 48, "tags": {"travel", "history", "finance", "tech", "reading"}},
    {"id": 12, "age": 31, "tags": {"tech", "music", "gaming", "food", "anime"}},
]

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
    return len(set1 & set2) / len(set1 | set2)


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

    return final_clusters,features, mlb



def visualize_clusters(usrs, features, mlb):
    # Get clusters
    clusters, features, mlb = get_clusters(usrs)

    # Use PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Set up colors for clusters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    plt.figure(figsize=(12, 8))

    # Plot each cluster
    for i, cluster in enumerate(clusters):
        # Find indices of users in this cluster
        cluster_indices = []
        for user in cluster:
            for j, u in enumerate(usrs):
                if u["id"] == user["id"]:
                    cluster_indices.append(j)
                    break

        # Extract coordinates for these users
        x = features_2d[cluster_indices, 0]
        y = features_2d[cluster_indices, 1]

        # Plot the cluster
        color = colors[i % len(colors)]
        plt.scatter(x, y, s=100, c=color, alpha=0.7, edgecolors='k', label=f'Cluster {i + 1}')

        # Add user ID labels
        for idx, x_val, y_val in zip(cluster_indices, x, y):
            plt.annotate(f'U{usrs[idx]["id"]}',
                         (x_val, y_val),
                         textcoords="offset points",
                         xytext=(0, 5),
                         ha='center')

    # Add information about each user in a text box
    user_info = "\n".join([f"User {u['id']}: Age {u['age']}, Tags: {', '.join(u['tags'])}" for u in usrs])
    plt.figtext(1.02, 0.5, user_info, fontsize=9, verticalalignment='center')

    # Add some statistics about the clusters
    cluster_stats = []
    for i, cluster in enumerate(clusters):
        ages = [user["age"] for user in cluster]
        avg_age = sum(ages) / len(ages) if ages else 0
        cluster_stats.append(f"Cluster {i + 1}: {len(cluster)} users, Avg age: {avg_age:.1f}")

    stats_text = "\n".join(cluster_stats)
    plt.figtext(0.02, 0.02, stats_text, fontsize=10)

    # Add visualization of feature importance
    feature_names = mlb.classes_.tolist() + ['Age']
    pca_components = np.vstack([pca.components_, np.zeros(len(feature_names))])

    plt.title('User Clusters with is_valid Constraints', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to make room for text
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig('user_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()

    return clusters


# Run the visualization
if __name__ == "__main__":
    print(get_clusters(users))
    visualize_clusters(users, None, None)
