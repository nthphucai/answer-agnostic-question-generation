import copy
import gc
import itertools
import os
import time

import pandas as pd
import torch
from loguru import logger

from celery import Celery
from celery.utils.log import get_task_logger
from mongodb.mongo_client import connect_mongodb
from questgen.pipelines.multitask_pipeline import MultiTaskPipeline
from questgen.pipelines.pipeline import pipeline
from questgen.utils.file_utils import read_yaml_file
from questgen.utils.utils import multiprocess
from restapi.schema.context_answer import Context


celery_logger = get_task_logger(__name__)

BROKER_URL = "redis://103.119.132.170:6373"
REDIS_URL = "rpc://admin:admin@103.119.132.170:5677"
app = Celery("tasks", broker=BROKER_URL, backend=REDIS_URL)

PIPELINE_CONFIG_PATH = os.getenv(
    "PIPELINE_CONFIG_PATH", "configs/faqg_pipeline_t5_vi_base_hl.yaml"
)

# Read the configuration for question generation pipeline
config = read_yaml_file(PIPELINE_CONFIG_PATH)

multi_task_pipeline = pipeline(
    "multitask",
    model="output/models/english/fschool_english_simple_questgen_v1.0",
    config_path=PIPELINE_CONFIG_PATH,
)

multiple_choice_pipeline = pipeline(
    "multiplechoice",
    model="output/models/english/fschool_english_multiple_questgen_v1.0",
    config_path=PIPELINE_CONFIG_PATH,
)

generate_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "generate_collection")


@app.task()
def generate(context: Context):
    logger.info("Got Request - Starting work ")
    return using_model(context)


def using_model(context: Context):
    try:
        logger.info("Generating questions from a context")
        context_text = context.context
        start_time = time.time()

        logger.info(context_text)
        result = multi_task_pipeline(
            [{"org_context": context_text, "context": context_text}]
        )
        generated_qa = multiprocess(
            MultiTaskPipeline.mapping_qa_context,
            range(len(result[0])),
            workers=2,
            result=result,
        )
        generated_qa = pd.DataFrame(itertools.chain(*generated_qa)).to_dict("records")
        logger.info(generated_qa)

        if config["type"] == "mc":
            for idc, _ in enumerate(generated_qa):
                generated_qa[idc]["context"] = generated_qa[idc]["org_context"]
                generated_qa[idc]["question"] = [generated_qa[idc]["question"]]
                generated_qa[idc]["options"] = [generated_qa[idc]["answer"]]
                generated_qa[idc]["answers"] = "A"

            generated_qa = multiple_choice_pipeline(generated_qa)

        elapsed_time = round(time.time() - start_time, 3)

        result = {
            "context": context_text,
            "results": generated_qa,
            "time": elapsed_time,
        }
        log_item = copy.deepcopy(result)
        generate_mongodb.insert_one(log_item)
        logger.info(f"Ready to return: {result}")

        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        logger.exception(e)
        return None
