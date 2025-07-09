from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Body
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select, func
import logging

from src import schemas, enums, models, crud, utils, utils_base, exceptions
from src.core.middleware import get_current_user
from src.database import get_session
from src.config.settings import settings
import src.config.subscription_plans as plans

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/subscription", tags=["Subscription"])


class SubscriptionPlansResponse(BaseModel):
    all: list[schemas.SubscriptionPlanExtended]

class PromoCodeActivateResponse(BaseModel):
    message: str

@router.get("/list", response_model=SubscriptionPlansResponse)
async def get_subscription_plans():
    return SubscriptionPlansResponse(all=plans.subscription_plans)


@router.post("/check", response_model=schemas.CurrentSubscribtion)
async def get_user_subscribtion(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Отдельный эндпоинт, который можно использовать для перепроверки баланса
    """
    new_subscribtion_id = await utils.check_user_subscription(user, db)
    user_plan = plans.get_subscription_plan(new_subscribtion_id)
    crud.create_user_plan_check(db, user.id, new_subscribtion_id)
    credits_left = max(0, user_plan.max_credits - user.used_credits_today)
    return schemas.CurrentSubscribtion(subscription_id=new_subscribtion_id, credits_left=credits_left)


@router.post("/check/force", response_model=schemas.CurrentSubscribtion)
async def get_user_subscribtion_force(user: models.User = Depends(get_current_user), db: Session = Depends(get_session)):
    """
    Отдельный эндпоинт, который можно использовать для перепроверки баланса
    """
    new_subscribtion_id = await utils.check_user_access(user)
    logger.info(f"New subscription id: {new_subscribtion_id}")
    user_plan = plans.get_subscription_plan(new_subscribtion_id)
    
    crud.create_user_plan_check(db, user.id, new_subscribtion_id)
    credits_left = max(0, user_plan.max_credits - user.used_credits_today)
    return schemas.CurrentSubscribtion(subscription_id=new_subscribtion_id, credits_left=credits_left)


@router.post("/promo/activate", response_model=PromoCodeActivateResponse)
async def activate_promo_code(
    request: schemas.PromoCodeActivateRequest = Body(...),
    session: Session = Depends(get_session),
    user: models.User = Depends(get_current_user),
):
    code = utils_base.format_promo_code(request.code)
    
    if user.pro_plan_promo_activated:
        raise exceptions.PromoCodeActivationError(message="You already have an endless PRO plan.")
    
    await crud.activate_promo_code(session, user, code)
    
    return PromoCodeActivateResponse(message="Promo code successfully activated!")
