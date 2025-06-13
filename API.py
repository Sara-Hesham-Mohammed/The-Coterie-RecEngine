import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from GraphRec.graphrec_fixed import GraphRec
from models.models import RecommendationRequest, RecommendationResponse
import recommend
from models.models import Event, User

app = FastAPI()

# Configure device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REPO = 'https://huggingface.co/foureyedpookie/GraphRec-Trained'


model_path = hf_hub_download(
    repo_id="foureyedpookie/GraphRec-Trained",
    filename="graphrec_meetup.pth"
)

print("Model downloaded to:", model_path)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
async def root():
    return {"message": "This is the python backend for the recommendation system."}




#### RECOMMENDATION ENDPOINT ####
@app.post("/get-recommendation")
async def get_recommendation(request: RecommendationRequest):
    try:

        user = request.user
        candidate_events = request.candidate_events
        model = GraphRec()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Initialize recommender
        recommender = recommend.EventRecommender(model_path)


        # Get recommendations for the new user
        recommended_indices, scores = recommender.get_recommendations(user, candidate_events)

        print("Recommendations for new user:")
        recommendations = []
        for idx, score in zip(recommended_indices, scores):
            event = candidate_events[idx]
            print(f"Event: {event.event_name}, Score: {score:.3f}")
            recommendations.append({
                "event": event,
                "score": float(score)  # Convert numpy float to Python float
            })

        response = RecommendationResponse(recommendations=recommendations)
        return {"Prediction": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



def main():
    recommender = recommend.EventRecommender(model_path)

    # Example: Create a new user
    new_user = User(
        user_id=999999,  # Use a large ID to avoid conflicts
        location="New York, NY",
        tags={"technology", "networking", "AI"},
        history_events=[]  # No history for new user
    )

    # Example: Create some new events
    new_events = [
        Event(
            event_name="AI Tech Meetup",
            location="New York, NY",
            tags={"technology", "networking", "AI"},
            is_paid=False
        ),
        Event(
            event_name="Food Festival",
            location="Brooklyn, NY",
            tags={"social", "food", "festival"},
            is_paid=True
        ),
    ]

    # Get recommendations for the new user
    recommended_indices, scores = recommender.get_recommendations(new_user, new_events)

    print("Recommendations for new user:")
    for idx, score in zip(recommended_indices, scores):
        print(f"Event: {new_events[idx].event_name}, Score: {score:.3f}")


if __name__ == "__main__":
    main()