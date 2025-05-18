from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    model = 5 #TODO: Load model here
    return model

#### RECOMMENDATION ENDPOINT ####
@app.post("/get-recommendation")
async def get_recommendation(request: RecommendationRequest):
    try:
        model = get_model("model.pth")
        user = request.user
        all_events = request.all_events
        prediction = model.predict(user.user_id, user.attendedEvents, all_events)
        return {"Prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))