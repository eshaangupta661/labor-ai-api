from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_data = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

model = model_data["model"]
model_columns = model_data["columns"]

class WageRequest(BaseModel):
    state: str
    sector: str
    experience_years: int
    skill_level: int
    offered_wage: float

@app.post("/predict")
def predict_wage(request: WageRequest):

    input_dict = {
        "experience_years": request.experience_years,
        "skill_level": request.skill_level
    }

    # Add one-hot columns
    for col in model_columns:
        if col.startswith("state_") or col.startswith("sector_"):
            input_dict[col] = 0

    state_col = f"state_{request.state}"
    sector_col = f"sector_{request.sector}"

    if state_col in input_dict:
        input_dict[state_col] = 1

    if sector_col in input_dict:
        input_dict[sector_col] = 1

    input_df = pd.DataFrame([input_dict])

    predicted_wage = model.predict(input_df)[0]

    status = "Underpaid" if request.offered_wage < 0.8 * predicted_wage else "Fair"

    return {
        "predicted_wage": round(float(predicted_wage), 2),
        "fairness_status": status
    }