from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Body
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func

from src import schemas, enums, models, crud, utils, utils_base
from src.core.middleware import get_current_user, check_balance_and_update_token
from src.database import get_session
from src.config.settings import settings
from src.config.subscription_plans import subscription_plans

router = APIRouter(prefix="/subscription", tags=["Subscription"])


class CurrentSubscribtion(BaseModel):
    subscription_id: enums.SubscriptionPlanType

class SubscriptionPlansResponse(BaseModel):
    all: list[schemas.SubscriptionPlanExtended]

class PromoCodeActivateResponse(BaseModel):
    message: str

@router.get("/list", response_model=SubscriptionPlansResponse)
async def get_subscription_plans():
    return SubscriptionPlansResponse(all=subscription_plans)

@router.post("/check", response_model=CurrentSubscribtion)
async def get_user_subscribtion(subscribtion: enums.SubscriptionPlanType = Depends(check_balance_and_update_token)):
    """
    Отдельный эндпоинт, который можно использовать для перепроверки баланса
    
    Нужно сбрасывать текущий токен, так как 
    """
    # TODO: уязвимое место для DoS атак
    return CurrentSubscribtion(subscription_id=subscribtion)

@router.post("/promo/activate", response_model=PromoCodeActivateResponse)
async def activate_promo_code(
    request: schemas.PromoCodeActivateRequest = Body(...),
    session: Session = Depends(get_session),
    user: models.User = Depends(get_current_user),
):
    code = utils_base.format_promo_code(request.code)
    
    if user.pro_plan_promo_activated:
        return PromoCodeActivateResponse(message="You already have an endless PRO plan.")
    
    await crud.activate_promo_code(session, user, code)
    
    return PromoCodeActivateResponse(message="Promo code successfully activated!")
