import os
from datetime import datetime

import fastapi
from loguru import logger

from mongodb.mongo_client import connect_mongodb
from restapi.schema.context_answer import FeedBack, UserFeedback


CONFIG_PATH = os.getenv(
    "API_FEEDBACK_CONFIG_PATH", "configs/faqg_pipeline_t5_vi_base_hl.yaml"
)
feedback_mongodb = connect_mongodb(CONFIG_PATH, "feedback_collection")
response_mongodb = connect_mongodb(CONFIG_PATH, "response_collection")

router = fastapi.APIRouter()


@router.get("/v1/feedback/healthcheck")
async def healthcheck():
    return {"message": "OK"}


@router.post("/v1/feedback/webapp", include_in_schema=False, status_code=200)
async def feedback(context: FeedBack):
    if feedback_mongodb is not None:
        try:
            result = {
                "task": context.task,
                "domain": context.domain,
                "results": context.results,
                "time": context.time,
                "log_time": str(datetime.now()),
            }
            feedback_mongodb.insert_one(result)
            return {"message": "feedback success"}

        except Exception as e:
            logger.exception(e)
            return {"error": "validation error"}
    else:
        logger.error("Can not connect to database")
        return {"error": "something wrong with server"}


@router.post("/v1/feedback/user", include_in_schema=False, status_code=200)
async def user_feedback(feedback: UserFeedback):
    if response_mongodb is not None:
        try:
            result = {
                "task": feedback.task,
                "domain": feedback.domain,
                "data": feedback.data,
                "time": feedback.time,
                "label": feedback.label,
                "rating": feedback.rating,
                "comment": feedback.comment,
                "log_time": str(datetime.now()),
            }
            response_mongodb.insert_one(result)
            return {"message": "feedback success"}
        except Exception as e:
            logger.error(e)
            return {"error": "validation error"}
    else:
        logger.error("Can not connect to database")
        return {"error": "something wrong with server"}
