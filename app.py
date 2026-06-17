from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run
from typing import Optional

from src.constants import APP_HOST, APP_PORT
from src.pipeline.prediction_pipeline import (
    PredictiveMaintenance,
    PredictionMaintenanceClassifier
)
from src.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DataForm:

    def __init__(self, request: Request):
        self.request = request

        self.Air_temperature = None
        self.Process_temperature = None
        self.Rotational_speed = None
        self.Torque = None
        self.Tool_Wear = None
        self.TypeEncoder = None

    async def get_predictive_data(self):
        form = await self.request.form()

        self.Air_temperature = float(form.get("Air_temperature"))
        self.Process_temperature = float(form.get("Process_temperature"))
        self.Rotational_speed = int(form.get("Rotational_speed"))
        self.Torque = float(form.get("Torque"))
        self.Tool_Wear = int(form.get("Tool_Wear"))
        self.TypeEncoder = int(form.get("TypeEncoder"))


@app.get("/", tags = "authentication")
async def index(request: Request):
    return templates.TemplateResponse(
    name="index.html",
    context={
        "request": request,
        "context": "Rendering"
    }
)


# @app.get("/train")
# async def trainRouteClient():
# 
#     try:
#         train_pipeline = TrainPipeline()
#         train_pipeline.run_pipeline()
# 
#         return Response("Training Successful")
# 
#     except Exception as e:
#         return Response(f"Error : {e}")


@app.post("/")
async def predictRouteClient(request: Request):

    try:

        form = DataForm(request)
        await form.get_predictive_data()

        predictive_data = PredictiveMaintenance(
            Air_temperature=form.Air_temperature,
            Process_temperature=form.Process_temperature,
            Rotational_speed=form.Rotational_speed,
            Torque=form.Torque,
            Tool_Wear=form.Tool_Wear,
            TypeEncoder=form.TypeEncoder
        )

        input_df = predictive_data.get_predictive_data_as_data_frame()

        predictor = PredictionMaintenanceClassifier()

        prediction = predictor.predict(input_df)[0]

        status = "Machine Failure" if prediction == 1 else "No Machine Failure"

        return templates.TemplateResponse(
        name = "index.html",
        context = {
            "request": request,
            "context": status
        }
    )

    except Exception as e:
        return {"status": False, "error": str(e)}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)