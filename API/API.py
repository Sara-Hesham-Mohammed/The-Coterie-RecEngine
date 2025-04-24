import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


def get_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
        return model

app = FastAPI()
#to be able to send requests using html file?
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

#send user info for prediction?
@app.post("/recommendation/")
async def get_recommendation():
    # PASS THE USER STUFF IN THE PARAMS
    # call the graph rec model here?
    model = get_model("models/recommendation.pkl")
    user = 'smth'
    #prediction = model.predict(user)
    prediction = ['x','y','z']
    return {"Prediction": f"Get user info and return the recommendations for that user, prediction is {prediction[1]}"}

@app.post("/recommendation/")
async def get_group():
