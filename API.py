import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torch import device

from GraphRec_WWW19.graphrec_fixed import GraphRec
from models.models import RecommendationRequest

app = FastAPI()

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


def get_model(path, args=None, kwargs=None):
    model = GraphRec(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

#### RECOMMENDATION ENDPOINT ####
@app.post("/get-recommendation")
async def get_recommendation(request: RecommendationRequest):
    try:
        model = get_model("model.pth")
        user = request.user
        all_events = request.all_events

        prediction = model.get_recommended_events(
            model,
            user.user_id,
            all_events,
            device,
            top_k=10,
            history_u_lists=user.attendedEvents
        )
        return {"Prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))