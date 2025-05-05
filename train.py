import torch
from GraphRec_WWW19.graphrec_fixed import load_model_for_inference, add_new_user, update_user_event_interaction, \
    get_recommended_events
import requests


class APIBasedGraphRec:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the initial model
        self.model, self.history_u_lists, self.history_v_lists, self.ratings_list = load_model_for_inference(
            model_path, self.device)

        # Keep track of API data
        self.social_adj_lists = {}  # We'll need to fetch this from API
        self.history_ur_lists = {}  # We'll need to fetch this from API
        self.history_vr_lists = {}  # We'll need to fetch this from API

        # Cache for users and events we've seen
        self.user_cache = set()
        self.event_cache = set()

    def fetch_user_data(self, user_id):
        """Fetch user data from API and update our data structures"""
        # Example API call - replace with your actual API
        user_data = requests.get(f"https://api.example.com/users/{user_id}").json()

        # If this is a new user
        if user_id not in self.user_cache:
            # Get their social connections
            social_connections = user_data.get('connections', [])

            # Add the new user
            self.history_u_lists, self.history_ur_lists, self.social_adj_lists = add_new_user(
                user_id, social_connections,
                self.history_u_lists, self.history_ur_lists, self.social_adj_lists
            )

            self.user_cache.add(user_id)

        # Update their history based on API data
        interactions = user_data.get('interactions', [])
        for interaction in interactions:
            event_id = interaction['event_id']
            rating = interaction['rating']

            # Update the interaction data
            self.history_u_lists, self.history_ur_lists, self.history_v_lists, self.history_vr_lists = update_user_event_interaction(
                user_id, event_id, rating,
                self.history_u_lists, self.history_ur_lists,
                self.history_v_lists, self.history_vr_lists
            )

            self.event_cache.add(event_id)

    def get_recommendations(self, user_id, top_k=10):
        """Get recommendations for a user"""
        # First, fetch latest data for this user
        self.fetch_user_data(user_id)

        # Get all available events
        all_events = list(self.event_cache)

        # Get recommendations
        recommendations = get_recommended_events(
            self.model, user_id, all_events, self.device,
            top_k=top_k, history_u_lists=self.history_u_lists
        )

        return recommendations

    def handle_cold_start(self, user_id):
        """Handle cold start for new users"""
        # For new users, we could:
        # 1. Recommend popular events
        # 2. Use content-based recommendations
        # 3. Ask for initial preferences

        # Example: Recommend top 5 most popular events
        event_popularity = {}
        for event_id in self.history_v_lists:
            event_popularity[event_id] = len(self.history_v_lists[event_id])

        popular_events = sorted(event_popularity.items(), key=lambda x: x[1], reverse=True)[:5]

        return [(event_id, 5.0) for event_id, _ in popular_events]  # Assume max rating of 5.0
