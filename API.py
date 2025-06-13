from models.models import User, RecommendationRequest, RecommendationResponse, Event
from Recommender import Recommender
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
import recommend


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
async def get_recommendation(new_user: User, candidate_events: list[Event]):
    try:
        request = RecommendationRequest(
            user=new_user,
            candidate_events=candidate_events
        )

        # Initialize service (replace with your actual values)
        service = Recommender(
            model_path=model_path,
            num_users=1000,  # Your actual number
            num_items=500  # Your actual number
        )

        # Get recommendations
        response = service.get_recommendations(request)

        print("Recommendations:")
        for i, rec in enumerate(response.recommendations):
            print(f"{i + 1}. {rec['event_name']} - Score: {rec['predicted_rating']:.2f}")

        return {"Prediction": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():

    new_user = User(
        user_id=99999,  # New user ID not in training
        location="San Francisco",
        tags={"tech", "networking"},
        history_events=[
            Event(
                event_name="Tech Meetup",
                location="San Francisco",
                tags={"tech", "AI"},
                is_paid=False
            )
        ]
    )

    candidate_events = [
        Event(
            event_name="AI Conference",
            location="San Francisco",
            tags={"AI", "tech"},
            is_paid=True
        ),
        Event(
            event_name="Cooking Class",
            location="Oakland",
            tags={"cooking", "social"},
            is_paid=True
        )
    ]

    request = RecommendationRequest(
        user=new_user,
        candidate_events=candidate_events
    )

    # Initialize service (replace with your actual values)
    service = Recommender(
        model_path=model_path,
        num_users=1000,  # Your actual number
        num_items=500  # Your actual number
    )

    # Get recommendations
    response = service.get_recommendations(request)

    print("Recommendations:")
    for i, rec in enumerate(response.recommendations):
        print(f"{i + 1}. {rec['event_name']} - Score: {rec['predicted_rating']:.2f}")


if __name__ == "__main__":
    main()