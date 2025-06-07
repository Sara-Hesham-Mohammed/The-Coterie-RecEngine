import torch
import pickle
from GraphRec import Social_Aggregators as soc_agg, UV_Encoders as uv_enc, graphrec_fixed as model, Social_Encoders as soc_enc, UV_Aggregators as uv_ag


class EventRecommender:
    def __init__(self, model_path, data_path, device=None):
        """
        Initialize the recommender with a trained model and necessary data
        
        Args:
            model_path (str): Path to the saved model file
            data_path (str): Path to the pickle file containing necessary data structures
            device (torch.device): Device to run the model on (CPU/GPU)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the saved model and data
        self.load_model_and_data(model_path, data_path)
        
    def load_model_and_data(self, model_path, data_path):
        """Load the trained model and required data structures"""
        # Load model metadata
        metadata = torch.load(model_path, map_location=self.device)
        
        # Load data structures
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.history_u_lists = data[0]
            self.history_ur_lists = data[1]
            self.history_v_lists = data[2]
            self.history_vr_lists = data[3]
            self.social_adj_lists = data[10]
            self.ratings_list = data[11]
        
        # Get dimensions
        self.num_users = len(self.history_u_lists)
        self.num_items = len(self.history_v_lists)
        self.num_ratings = len(self.ratings_list)
        self.embed_dim = metadata['embed_dim']
        
        # Initialize embeddings
        self.u2e = torch.nn.Embedding(self.num_users, self.embed_dim).to(self.device)
        self.v2e = torch.nn.Embedding(self.num_items, self.embed_dim).to(self.device)
        self.r2e = torch.nn.Embedding(self.num_ratings, self.embed_dim).to(self.device)
        
        # Create model components
        agg_u_history = uv_ag.UV_Aggregator(self.v2e, self.r2e, self.u2e, self.embed_dim, cuda=self.device, uv=True)
        enc_u_history = uv_enc.UV_Encoder(self.u2e, self.embed_dim, self.history_u_lists, self.history_ur_lists,
                                 agg_u_history, cuda=self.device, uv=True)
        
        agg_u_social = soc_agg.Social_Aggregator(lambda nodes: enc_u_history(nodes).t(),
                                       self.u2e, self.embed_dim, cuda=self.device)
        enc_u = soc_enc.Social_Encoder(lambda nodes: enc_u_history(nodes).t(), self.embed_dim,
                             self.social_adj_lists, agg_u_social, base_model=enc_u_history, 
                             cuda=self.device)
        
        agg_v_history = uv_ag.UV_Aggregator(self.v2e, self.r2e, self.u2e, self.embed_dim, cuda=self.device, uv=False)
        enc_v_history = uv_enc.UV_Encoder(self.v2e, self.embed_dim, self.history_v_lists, self.history_vr_lists,
                                 agg_v_history, cuda=self.device, uv=False)
        
        # Create and load the model
        self.model = model.GraphRec(enc_u, enc_v_history, self.r2e).to(self.device)
        self.model.load_state_dict(metadata['model_state'])
        self.model.eval()

        # Save the embeddings for similarity calculations
        with torch.no_grad():
            self.user_embeddings = self.u2e.weight.data.clone()
            self.event_embeddings = self.v2e.weight.data.clone()

    def get_user_embedding(self, user_features):
        """
        Generate embedding for a new user based on their features and similarity to existing users
        
        Args:
            user_features (dict): Dictionary containing user features like:
                - interests (list): List of user interests/tags
                - location (tuple): (latitude, longitude)
                - age (int): User age
                - etc.
                
        Returns:
            torch.Tensor: Embedding vector for the new user
        """
        # Convert user features into a feature vector
        feature_vector = torch.zeros(self.embed_dim).to(self.device)
        
        # TODO: Implement proper user feature processing
        # For now using a simple approach - find similar users in training data
        # based on interests/tags overlap and location proximity
        similar_users = []
        similarity_scores = []
        
        # For demonstration, using random initialization
        # In production, implement proper similarity search based on features
        similar_users = torch.randint(0, self.num_users, (5,))
        similarity_scores = torch.ones(5) / 5
        
        # Get embeddings of similar users
        similar_embeddings = self.user_embeddings[similar_users]
        
        # Compute new user embedding as weighted average of similar users
        similarity_scores = similarity_scores.to(self.device)
        new_embedding = (similar_embeddings * similarity_scores.unsqueeze(1)).sum(dim=0)
        
        return new_embedding
        
    def get_event_embedding(self, event_features):
        """
        Generate embedding for a new event based on its features and similarity to existing events
        
        Args:
            event_features (dict): Dictionary containing event features like:
                - tags (list): List of event tags
                - description (str): Event description
                - location (tuple): (latitude, longitude)
                - etc.
                
        Returns:
            torch.Tensor: Embedding vector for the new event
        """
        # Convert event features into a feature vector
        feature_vector = torch.zeros(self.embed_dim).to(self.device)
        
        # TODO: Implement proper event feature processing
        # For now using a simple approach - find similar events in training data
        # based on tag overlap and location proximity
        similar_events = []
        similarity_scores = []
        
        # For demonstration, using random initialization
        # In production, implement proper similarity search based on features
        similar_events = torch.randint(0, self.num_items, (5,))
        similarity_scores = torch.ones(5) / 5
        
        # Get embeddings of similar events
        similar_embeddings = self.event_embeddings[similar_events]
        
        # Compute new event embedding as weighted average of similar events
        similarity_scores = similarity_scores.to(self.device)
        new_embedding = (similar_embeddings * similarity_scores.unsqueeze(1)).sum(dim=0)
        
        return new_embedding

    def get_recommendations(self, user_features, candidate_events_features, top_k=5):
        """
        Get recommendations for a new user from a list of new candidate events
        
        Args:
            user_features (dict): Dictionary containing features of the new user
            candidate_events_features (list): List of dictionaries containing features for new events
            top_k (int): Number of top recommendations to return
            
        Returns:
            tuple: (recommended_events_indices, scores) - Indices of recommended events and their scores
        """
        with torch.no_grad():
            # Generate embedding for the new user
            user_embedding = self.get_user_embedding(user_features)
            
            # Generate embeddings for new events
            event_embeddings = []
            for event_features in candidate_events_features:
                embedding = self.get_event_embedding(event_features)
                event_embeddings.append(embedding)
            
            event_embeddings = torch.stack(event_embeddings)
            
            # Create temporary IDs
            temp_user_id = self.num_users  # Use an ID outside training range
            temp_event_ids = list(range(self.num_items, self.num_items + len(candidate_events_features)))
            
            # Prepare input tensors
            user_tensor = torch.LongTensor([temp_user_id] * len(temp_event_ids)).to(self.device)
            event_tensor = torch.LongTensor(temp_event_ids).to(self.device)
            
            # Temporarily extend the embedding layers
            original_user_weight = self.u2e.weight.data.clone()
            original_event_weight = self.v2e.weight.data.clone()
            
            self.u2e.weight.data = torch.cat([
                original_user_weight,
                user_embedding.unsqueeze(0)
            ])
            
            self.v2e.weight.data = torch.cat([
                original_event_weight,
                event_embeddings
            ])
            
            # Get predictions
            scores = self.model(user_tensor, event_tensor)
            
            # Restore original embeddings
            self.u2e.weight.data = original_user_weight
            self.v2e.weight.data = original_event_weight
            
            # Get top-k recommendations
            top_scores, indices = torch.topk(scores, min(top_k, len(temp_event_ids)))
            
            recommended_indices = indices.cpu().numpy()
            recommendation_scores = top_scores.cpu().numpy()
            
            return recommended_indices, recommendation_scores


def main():
    # Example usage
    model_path = "graphrec_meetup.pth"
    data_path = "Pre-processing/clean_meetup_data.pickle"
    
    # Initialize recommender
    recommender = EventRecommender(model_path, data_path)
    
    # Example: Create a new user with features
    new_user = {
        "interests": ["technology", "networking", "AI"],
        "location": (40.7128, -74.0060),  # NYC coordinates
        "age": 28
    }
    
    # Example: Create some new events with features
    new_events = [
        {
            "tags": ["technology", "networking"],
            "description": "A tech meetup about AI",
            "location": (40.7128, -74.0060)  # NYC coordinates
        },
        {
            "tags": ["social", "food"],
            "description": "Food festival with local vendors",
            "location": (40.7214, -73.9951)
        },
        # Add more events...
    ]
    
    # Get recommendations for the new user
    recommended_indices, scores = recommender.get_recommendations(new_user, new_events)
    
    print("Recommendations for new user:")
    for idx, score in zip(recommended_indices, scores):
        print(f"Event {idx} (features: {new_events[idx]}): Score {score:.3f}")


if __name__ == "__main__":
    main() 