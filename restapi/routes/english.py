import copy
import os
import random
import sys

import fastapi
import torch
from fastapi import File, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from loguru import logger

from questgen.inference import Inference
from restapi.utils import file_preprocess, generate, split_context


sys.path.append("utils")

from mongodb.mongo_client import connect_mongodb
from questgen.dataset.create_data.modules.preprocessor import Preprocessor
from questgen.utils.file_utils import read_yaml_file
from restapi.schema.context_answer import QuestgenRequestItem


router = fastapi.APIRouter()

PIPELINE_CONFIG_PATH = os.getenv(
    "PIPELINE_CONFIG_PATH", "configs/faqg_pipeline_t5_vi_base_hl.yaml"
)

SIMPLE_MODEL = os.getenv("SIMPLE_MODEL")
MC_MODEL = os.getenv("MC_MODEL")

random.seed(1358)
torch.manual_seed(1358)
# Read the configuration for question generation pipeline
config = read_yaml_file(PIPELINE_CONFIG_PATH)

model = Inference(
    config_aqg_path=PIPELINE_CONFIG_PATH,
    multitask_model_name_or_path=SIMPLE_MODEL,
    mc_model_name_or_path=MC_MODEL,
    tokenizer_name_or_path=SIMPLE_MODEL,
)

generate_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "generate_collection")
feedback_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "feedback_collection")


@router.get("/v1/english/healthcheck")
async def healthcheck():
    return {"message": "OK"}


@router.get("/v1/english/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/v1/english/openapi.json",
        title="API",
    )


@router.post("/v1/english/generate_from_file")
async def upload_file(task: str = "", file: UploadFile = File(...)):
    raw_context = file_preprocess(file)
    preprocessor = Preprocessor("configs/parse_pdf_config.yml")
    context = preprocessor.preprocess_pdf_context([raw_context])
    context_text = split_context(context[0])

    try:
        result = generate(context_text=context_text, model=model, task=task, lang="eng")
        log_item = copy.deepcopy(result)
        generate_mongodb.insert_one(log_item)
        return result
    except Exception as e:
        logger.exception(e)
        return None


@router.post("/v1/english/generate")
async def generate_qa(request_item: QuestgenRequestItem):
    try:
        logger.info("Generating questions from a context")
        context_text = request_item.context
        context_text = split_context(context_text)
        result = generate(
            context_text=context_text, model=model, task=request_item.task, lang="eng"
        )

        log_item = copy.deepcopy(result)
        generate_mongodb.insert_one(log_item)
        logger.info(f"Ready to return: {result}")
    except Exception as e:
        logger.exception(e)
        return None
    return result
