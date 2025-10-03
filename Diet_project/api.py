from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List

app = FastAPI()
df = pd.read_csv("clustered_food_data.csv")

class Input(BaseModel):
    age: int
    height: float
    weight: float
    food_pref: str

@app.post("/recommend")
def recommend(data: Input):
    height_m = data.height / 100
    bmi = data.weight / (height_m ** 2)

    if bmi < 18.5:
        category, cluster = "Underweight", "Weight Gain"
    elif bmi < 25:
        category, cluster = "Healthy Weight", "Healthy"
    else:
        category, cluster = "Overweight", "Weight Loss"

    filtered = df[(df["cluster_label"] == cluster) & (df["food_type"] == data.food_pref)]
    if len(filtered) < 9:
        return {"error": "Not enough food items"}

    selected = filtered.sample(9).reset_index(drop=True)
    return {
        "bmi": round(bmi, 2),
        "category": category,
        "cluster": cluster,
        "breakfast": selected.iloc[0:3].to_dict(orient="records"),
        "lunch": selected.iloc[3:6].to_dict(orient="records"),
        "dinner": selected.iloc[6:9].to_dict(orient="records"),
    }
