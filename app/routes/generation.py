import os

import fastapi
import requests
from loguru import logger

from app.schema.context_answer import Context, FeedBack, FIBContext, HistoryTextbookContext


router = fastapi.APIRouter()

GENERATE_ENDPOINT = os.getenv(
    "GENERATE_ENDPOINT", "http://103.119.132.170:5050/api/v1/generate"
)
FEEDBACK_ENDPOINT = os.getenv(
    "FEEDBACK_ENDPOINT", "http://103.119.132.170:5050/api/v1/feedback"
)
GENERATE_FIB_ENDPOINT = os.getenv(
    "GENERATE_FIB_ENDPOINT", "http://103.119.132.170:5050/api/v1/generate_fib"
)
GENERATE_HISTORY_TEXTBOOK_ENDPOINT = os.getenv(
    "GENERATE_HISTORY_TEXTBOOK_ENDPOINT",
    "http://103.119.132.170:5050/api/v1/generate_from_book",
)


@router.post("/api/v1/generate", include_in_schema=False)
async def generate(context: Context):
    try:
        output = requests.post(
            GENERATE_ENDPOINT,
            json={
                "task": context.task,
                "domain": context.domain,
                "context": context.context,
            },
        )

        if output.status_code == 200:
            result = output.json()
        else:
            result = None

        return result

    except Exception as e:
        logger.exception(e)
        return None


@router.post("/api/v1/generate_fib", include_in_schema=False)
async def generate_fib(context: FIBContext):
    try:
        output = requests.post(
            GENERATE_FIB_ENDPOINT,
            json={"context": context.context, "num_blank": context.num_blank},
        )

        if output.status_code == 200:
            result = output.json()
        else:
            result = None

        return result

    except Exception as e:
        logger.exception(e)
        return None


@router.post("/api/v1/generate_from_book", include_in_schema=False)
async def generate_from_book(context: HistoryTextbookContext):
    try:
        output = requests.post(
            GENERATE_HISTORY_TEXTBOOK_ENDPOINT,
            json={"task": context.task, "section": context.section},
        )

        if output.status_code == 200:
            result = output.json()
        else:
            result = None

        return result

    except Exception as e:
        logger.exception(e)
        return None


@router.post("/api/v1/feedback", include_in_schema=False)
async def feedback(result: FeedBack):
    try:
        requests.post(
            FEEDBACK_ENDPOINT,
            json={
                "task": result.task,
                "domain": result.domain,
                "results": result.results,
                "time": result.time,
            },
        )

    except Exception as e:
        logger.error(e)
