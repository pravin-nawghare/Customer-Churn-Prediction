
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from churn_prediction.constants import APP_HOST, APP_PORT
from churn_prediction.pipeline.prediction_pipeline import CustomerChurnData, CustomerChurnClassifier
from churn_prediction.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request): #user wil give this data
        self.request: Request = request
        self.married: Optional[str] = None
        self.city: Optional[str] = None
        self.multiple_lines: Optional[str] = None
        self.number_of_dependents: Optional[str] = None
        self.number_of_referrals: Optional[str] = None
        self.device_protection_plan: Optional[str] = None
        self.total_revenue: Optional[str] = None
        self.online_security: Optional[str] = None
        self.total_extra_data_charges: Optional[str] = None
        self.payment_method: Optional[str] = None
        self.paperless_billing: Optional[str] = None
        self.internet_service: Optional[str] = None
        self.internet_type: Optional[str] = None
        self.online_backup: Optional[str] = None
        self.premium_tech_support: Optional[str] = None
        self.contract: Optional[str] = None
        self.refund_category: Optional[str] = None
        self.streaming_tv: Optional[str] = None
        self.streaming_movies: Optional[str] = None
        self.streaming_music: Optional[str] = None
        

    async def get_customer_data(self): # data collected from form
        form = await self.request.form()
        self.married = form.get("married")
        self.city = form.get("city")
        self.multiple_lines = form.get("multiple_lines")
        self.number_of_dependents = form.get("number_of_dependents")
        self.number_of_referrals = form.get("number_of_referrals")
        self.device_protection_plan = form.get("device_protection_plan")
        self.total_revenue = form.get("total_revenue")
        self.online_security = form.get("online_security")
        self.total_extra_data_charges = form.get("total_extra_data_charges")
        self.payment_method = form.get("payment_method")
        self.paperless_billing = form.get("paperless_billing")
        self.internet_service = form.get("internet_service")
        self.internet_type = form.get("internet_type")
        self.online_backup = form.get("online_backup")
        self.premium_tech_support = form.get("premium_tech_support")
        self.contract = form.get("contract")
        self.refund_category = form.get("refund_category")
        self.streaming_tv = form.get("streaming_tv")
        self.streaming_movies = form.get("streaming_movies")
        self.streaming_music = form.get("streaming_music")

@app.get("/", tags=["authentication"]) #landing/ main page
async def index(request: Request):

    return templates.TemplateResponse(
            "customer_data.html",{"request": request, "context": "Rendering"})


@app.get("/train") # start training
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/") # start prediction
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_customer_data()
        
        customer_data = CustomerChurnData(
                            married = form.married,
                            city = form.city,
                            multiple_lines = form.multiple_lines,
                            number_of_dependents = form.number_of_dependents,
                            number_of_referrals = form.number_of_referrals,
                            device_protection_plan = form.device_protection_plan,
                            total_revenue = form.total_revenue,
                            online_security = form.online_security,
                            total_extra_data_charges = form.total_extra_data_charges,
                            payment_method = form.payment_method,
                            paperless_billing = form.paperless_billing,
                            internet_service = form.internet_service,
                            internet_type = form.internet_type,
                            online_backup = form.online_backup,
                            premium_tech_support = form.premium_tech_support ,
                            contract = form.contract,
                            refund_category = form.refund_category,
                            streaming_tv = form.streaming_tv,
                            streaming_movies = form.streaming_music,
                            streaming_music = form.streaming_music
                            )
        
        churn_df = customer_data.get_customer_input_data_frame()

        model_predictor = CustomerChurnClassifier()

        value = model_predictor.predict(dataframe=churn_df)[0]

        status = None
        if value == 1:
            status = "Customer Will Leave"
        else:
            status = "Customer Will Stay"

        return templates.TemplateResponse(
            "customer_data.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)

'''
{"status":false,"error":"Error occured python script name 
[D:\\Data-Sorting\\us-visa-approval-machine-learning-project\\us_visa\\pipeline\\prediction_pipeline.py] 
line number [113] error message [Error occured python script name 
[D:\\Data-Sorting\\us-visa-approval-machine-learning-project\\us_visa\\entity\\s3_estimator.py] 
line number [62] error message [Error occured python script name 
[D:\\Data-Sorting\\us-visa-approval-machine-learning-project\\us_visa\\cloud_storage\\aws_storage.py] 
line number [124] error message [a bytes-like object is required, not 'function']]]"}
'''    