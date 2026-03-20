from fastapi import FastAPI, APIRouter
import joblib
from pydantic import BaseModel
import pandas as pd

# Load the model
xgb_model = joblib.load("../../models/housing_xgb_model.pkl")

# Define the input data xgb_model
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float
    rooms_per_household: float
    bedroom_per_room: float
    people_per_household: float
    income_per_household: float


# Create the FastAPI app
app = FastAPI()



@app.post("/predict/housing/xgboost")
def predict_housing_xgboost(data: HousingData):
        # Convert input to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Ensure column order matches training
        feature_order = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "rooms_per_household",
            "bedroom_per_room",
            "people_per_household",
            "income_per_household",
        ]

        input_data = input_data[feature_order]

        prediction = xgb_model.predict(input_data)

        return {"prediction": float(prediction[0])}
