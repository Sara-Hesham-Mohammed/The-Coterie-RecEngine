import random
import torch
from models.models import User, RecommendationRequest, RecommendationResponse, Event


class Recommender:
    """
    Simple approach: Use any existing user ID as proxy for new users
    """

    def __init__(self, model_path: str, num_users: int, num_items: int, device: str = 'cpu'):
        self.device = torch.device(device)
        self.num_users = num_users  # Total users in your training data
        self.num_items = num_items  # Total items in your training data

        # Load your trained model
        model_data = torch.load(model_path, map_location=self.device)
        # TODO: Initialize GraphRec model here
        # self.model = GraphRec(...)
        # self.model.load_state_dict(model_data['model_state'])
        # self.model.eval()
        self.model = None  # Placeholder - replace with your actual model

        print(f"Model loaded. Can use user IDs 0-{num_users - 1} and item IDs 0-{num_items - 1}")

    def get_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """
        Get recommendations for new user using a simple proxy approach
        """
        proxy_user_id = random.randint(0, self.num_users - 1)

        print(f"Using proxy user ID {proxy_user_id} for new user {request.user.user_id}")

        recommendations = []

        for event in request.candidate_events:
            # Map event to item ID
            item_id = self._map_event_to_item_id(event)

            # Make sure item_id is valid
            if item_id >= self.num_items:
                item_id = item_id % self.num_items  # Wrap around if too large

            # Create tensors for model input
            user_tensor = torch.tensor([proxy_user_id], dtype=torch.long).to(self.device)
            item_tensor = torch.tensor([item_id], dtype=torch.long).to(self.device)

            # Get prediction from existing model
            with torch.no_grad():
                if self.model:
                    predicted_rating = self.model.forward(user_tensor, item_tensor)
                    score = predicted_rating.item()
                else:
                    # Placeholder for demo
                    score = random.uniform(1.0, 5.0)

            recommendations.append({
                'event_name': event.event_name,
                'location': event.location,
                'tags': list(event.tags),
                'is_paid': event.is_paid,
                'predicted_rating': score,
                'proxy_user_id': proxy_user_id  # For debugging
            })

        # Sort by predicted rating (highest first)
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)

        return RecommendationResponse(recommendations=recommendations)

    def _map_event_to_item_id(self, event: Event) -> int:
        """
        Map an Event to item_id used in your model
        """
        # Simple hash-based mapping
        event_string = f"{event.event_name}_{event.location}"
        return abs(hash(event_string)) % self.num_items


# Usage example:
def create_recommendation_endpoint():

    NUM_USERS = 164463
    NUM_ITEMS = 234684

    service = Recommender(
        model_path='graphrec_model.pth',
        num_users=NUM_USERS,
        num_items=NUM_ITEMS
    )

    def recommend_events(request: RecommendationRequest) -> RecommendationResponse:
        return service.get_recommendations(request)

    return recommend_events