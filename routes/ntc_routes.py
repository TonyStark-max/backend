from fastapi import APIRouter

from schemas.ntc_schema import NTCApplication
from services.ntc_prediction_service import predict_ntc

router = APIRouter()


@router.post("/new-predict")
def new_predict(request: NTCApplication):
    return predict_ntc(request.model_dump())
