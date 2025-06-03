import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torch import device

from GraphRec_WWW19.graphrec_fixed import GraphRec
from models.models import RecommendationRequest

app = FastAPI()

# Configure device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def get_model(path):
    try:
        # Load the model state and metadata
        checkpoint = torch.load(path, map_location=DEVICE)
        
        # Initialize model with saved parameters
        model = GraphRec(
            enc_u=checkpoint['enc_u'],
            enc_v_history=checkpoint['enc_v_history'],
            r2e=checkpoint['r2e']
        )
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")

#### RECOMMENDATION ENDPOINT ####
@app.post("/get-recommendation")
async def get_recommendation(request: RecommendationRequest):
    try:
        model = get_model("models/trained_model.pth")
        user = request.user
        all_events = request.all_events

        prediction = model.get_recommended_events(
            model,
            user.user_id,
            all_events,
            DEVICE,
            top_k=10,
            history_u_lists=user.attendedEvents
        )
        return {"Prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))