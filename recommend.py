import torch
from models.models import Event, User
from typing import List

class EventRecommender:
    def __init__(self, model_path, device=None):
        """
        Initialize the recommender with a trained model
        
        Args:
            model_path (str): Path to the saved model file
            device (torch.device): Device to run the model on (CPU/GPU)
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the saved model
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained model"""
        # Load the complete saved model
        self.saved_model = torch.load(model_path, map_location=self.device)
        
        # Load the model
        self.model = self.saved_model['model']
        self.model.load_state_dict(self.saved_model['model_state'])
        self.model.to(self.device)
        self.model.eval()

        # Get the embedding layers from the loaded model
        self.u2e = self.model.user_embedding
        self.v2e = self.model.item_embedding
        self.r2e = self.model.rating_embedding

        # Get dimensions from the embedding layers
        self.num_users = self.u2e.weight.shape[0]
        self.num_items = self.v2e.weight.shape[0]
        self.embed_dim = self.u2e.weight.shape[1]

        # Save the embeddings for similarity calculations
        with torch.no_grad():
            self.user_embeddings = self.u2e.weight.data.clone()
            self.event_embeddings = self.v2e.weight.data.clone()

    def calculate_similarity_score(self, tags1: set[str], tags2: set[str], loc1: str, loc2: str) -> float:
        """
        Calculate similarity score between two entities based on their tags and location
        
        Args:
            tags1: Set of tags for first entity
            tags2: Set of tags for second entity
            loc1: Location of first entity
            loc2: Location of second entity
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Calculate tag similarity using Jaccard similarity
        tag_similarity = len(tags1.intersection(tags2)) / len(tags1.union(tags2)) if tags1 or tags2 else 0
        
        # Calculate location similarity (simple exact match for now)
        # TODO: Implement proper location similarity using coordinates
        loc_similarity = 1.0 if loc1 == loc2 else 0.0
        
        # Weighted combination (can be adjusted)
        return 0.7 * tag_similarity + 0.3 * loc_similarity

    def get_user_embedding(self, user: User) -> torch.Tensor:
        """
        Generate embedding for a new user based on their features
        
        Args:
            user: User model containing user features
                
        Returns:
            torch.Tensor: Embedding vector for the new user
        """
        # Find similar users based on tags and location
        similarity_scores = []
        
        # Compare with all users in training set
        # TODO: Optimize this using a proper similarity search index
        for i in range(self.num_users):
            # Get the embedding for user i and convert to features
            # This is a placeholder - in production you would have a mapping of user IDs to their features
            training_user_tags = {"placeholder"}  # Replace with actual tags from training data
            training_user_loc = "placeholder"     # Replace with actual location from training data
            
            score = self.calculate_similarity_score(
                user.tags,
                training_user_tags,
                user.location,
                training_user_loc
            )
            similarity_scores.append(score)
            
        # Convert to tensor
        similarity_scores = torch.tensor(similarity_scores, device=self.device)
        
        # Get top k similar users
        k = min(5, len(similarity_scores))
        top_scores, top_indices = torch.topk(similarity_scores, k)
        
        # Normalize scores to sum to 1
        top_scores = top_scores / top_scores.sum()
        
        # Get embeddings of similar users and compute weighted average
        similar_embeddings = self.user_embeddings[top_indices]
        new_embedding = (similar_embeddings * top_scores.unsqueeze(1)).sum(dim=0)
        
        return new_embedding
        
    def get_event_embedding(self, event: Event) -> torch.Tensor:
        """
        Generate embedding for a new event based on its features
        
        Args:
            event: Event model containing event features
                
        Returns:
            torch.Tensor: Embedding vector for the new event
        """
        # Find similar events based on tags and location
        similarity_scores = []
        
        # Compare with all events in training set
        # TODO: Optimize this using a proper similarity search index
        for i in range(self.num_items):
            # Get the embedding for event i and convert to features
            # This is a placeholder - in production you would have a mapping of event IDs to their features
            training_event_tags = {"placeholder"}  # Replace with actual tags from training data
            training_event_loc = "placeholder"     # Replace with actual location from training data
            
            score = self.calculate_similarity_score(
                event.tags,
                training_event_tags,
                event.location,
                training_event_loc
            )
            similarity_scores.append(score)
            
        # Convert to tensor
        similarity_scores = torch.tensor(similarity_scores, device=self.device)
        
        # Get top k similar events
        k = min(5, len(similarity_scores))
        top_scores, top_indices = torch.topk(similarity_scores, k)
        
        # Normalize scores to sum to 1
        top_scores = top_scores / top_scores.sum()
        
        # Get embeddings of similar events and compute weighted average
        similar_embeddings = self.event_embeddings[top_indices]
        new_embedding = (similar_embeddings * top_scores.unsqueeze(1)).sum(dim=0)
        
        return new_embedding

    def get_recommendations(self, user: User, candidate_events: List[Event], top_k: int = 5):
        """
        Get recommendations for a new user from a list of new candidate events
        
        Args:
            user: User model containing user features
            candidate_events: List of Event models containing event features
            top_k: Number of top recommendations to return
            
        Returns:
            tuple: (recommended_events_indices, scores) - Indices of recommended events and their scores
        """
        with torch.no_grad():
            # Generate embedding for the new user
            user_embedding = self.get_user_embedding(user)
            
            # Generate embeddings for new events
            event_embeddings = []
            for event in candidate_events:
                embedding = self.get_event_embedding(event)
                event_embeddings.append(embedding)
            
            event_embeddings = torch.stack(event_embeddings)
            
            # Create temporary IDs
            temp_user_id = self.num_users  # Use an ID outside training range
            temp_event_ids = list(range(self.num_items, self.num_items + len(candidate_events)))
            
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

