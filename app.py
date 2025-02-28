from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model.pkl")

# Initialize FastAPI
app = FastAPI()


# Define request body schema
class InputData(BaseModel):
    account_length: int
    area_code: int
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    customer_service_calls: int
    churn: bool


@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Convert categorical variables
        international_plan = 1 if data.international_plan.lower() == "yes" else 0
        voice_mail_plan = 1 if data.voice_mail_plan.lower() == "yes" else 0

        # Prepare the input for prediction
        input_data = np.array(
            [
                [
                    data.account_length,
                    data.area_code,
                    international_plan,
                    voice_mail_plan,
                    data.number_vmail_messages,
                    data.total_day_minutes,
                    data.total_day_calls,
                    data.total_day_charge,
                    data.total_eve_minutes,
                    data.total_eve_calls,
                    data.total_eve_charge,
                    data.total_night_minutes,
                    data.total_night_calls,
                    data.total_night_charge,
                    data.total_intl_minutes,
                    data.total_intl_calls,
                    data.total_intl_charge,
                    data.customer_service_calls,
                    data.churn,
                ]
            ]
        )

        # Debugging
        print(f"Input data: {input_data}")

        # Perform the prediction
        prediction = model.predict(input_data)

        # Convert prediction to human-readable result
        result = (
            "The customer will churn"
            if prediction[0]
            else "The customer will not churn"
        )

        return {"prediction": result}

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")
