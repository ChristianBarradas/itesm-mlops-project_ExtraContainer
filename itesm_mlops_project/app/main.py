import logging
import requests
from fastapi import FastAPI, Body
from custom_loggin import CustomLogger

FILE_LOG_DIR = 'frontend.log'

logger = CustomLogger(__name__, FILE_LOG_DIR).logger    
logger.info("start frontned FastAPI")
app = FastAPI()

# ML model prediction function using the prediction API request
def predict_house(input):
    url3 = "http://app.docker:8000/predict"

    response = requests.post(url3, json=input)
    response = response.text

    return response


@app.get("/")
def read_root():
    logger.info(f"Front-end is all ready to go!")
    return "Front-end is all ready to go!"

@app.post("/predict")
def classify(payload: dict = Body(...)):
    logger.debug(f"Incoming input in the front end: {payload}")
    response = predict_house(payload)
    logger.debug(f"Prediction: {response}")
    return {"response": response}


@app.get("/healthcheck")
async def v1_healhcheck():
    url3 = "http://app.docker:8000/"

    response = requests.request("GET", url3)
    response = response.text
    logger.info(f"Checking health: {response}")

    return response
