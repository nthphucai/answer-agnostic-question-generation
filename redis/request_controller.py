import json
import sys
import uuid

from loguru import logger

import aioredis


REDIS_URL = "redis://103.119.132.170"
REDIS_PORT = 6373


async def add_request_to_redis(request):
    try:
        redis = await aioredis.from_url(
            f"{REDIS_URL}:{REDIS_PORT}", encoding="utf-8", decode_responses=True
        )
        logger.success("Initialize redis successfully.")
    except Exception as e:
        logger.exception(e)
        sys.exit()

    try:
        logger.debug(f"Incoming request:\n {request}")
        key = str(uuid.uuid4())
        data = {"id": key, "request": request}
        await redis.rpush("REQUEST_QUEUE", json.dumps(data))
        logger.success("Add request to redis successfully.")
    except Exception as e:
        logger.exception(e)
        pass
