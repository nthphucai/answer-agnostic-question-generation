import copy
import gc
import os
import sys

from fastapi import File, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html

from questgen.dataset.create_data.modules.preprocessor import Preprocessor
from restapi.utils import file_preprocess, generate, split_context, transform_output


sys.path.append("utils")

from datetime import datetime

import fastapi
import torch
from loguru import logger

from mongodb.mongo_client import connect_mongodb
from questgen.inference import Inference
from questgen.utils.file_utils import read_yaml_file
from restapi.schema.context_answer import BookRequestItem, QuestgenRequestItem


router = fastapi.APIRouter()

PIPELINE_CONFIG_PATH = os.getenv(
    "PIPELINE_CONFIG_PATH", "configs/faqg_pipeline_t5_vi_base_hl.yaml"
)
SIMPLE_MODEL = os.getenv("SIMPLE_MODEL")
MC_MODEL = os.getenv("MC_MODEL")

config = read_yaml_file(PIPELINE_CONFIG_PATH)

model = Inference(
    config_aqg_path=PIPELINE_CONFIG_PATH,
    multitask_model_name_or_path=SIMPLE_MODEL,
    mc_model_name_or_path=MC_MODEL,
    tokenizer_name_or_path=SIMPLE_MODEL,
)

generate_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "generate_collection")
feedback_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "feedback_collection")


@router.get("/v1/history/healthcheck")
async def healthcheck():
    return {"message": "OK"}


@router.get("/v1/history/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/v1/history/openapi.json",
        title="API",
    )


@router.post("/v1/history/generate")
async def generate_qa(request_item: QuestgenRequestItem):
    try:
        logger.info("Generating questions from a context")
        context_text = request_item.context
        context_text = split_context(context_text)
        result = generate(
            context_text=context_text, model=model, task=request_item.task, lang="vi"
        )

        log_item = copy.deepcopy(result)
        generate_mongodb.insert_one(log_item)
        logger.info(f"Ready to return: {result}")
    except Exception as e:
        logger.exception(e)
        return None
    return result


@router.post("/v1/history/generate_from_book")
async def generate_from_book(request_item: BookRequestItem):
    logger.info("Generating questions from a context")
    path = request_item.section
    start_time = datetime.now()

    if request_item.task == "simple-question":
        qg_result = model.generate_qa_pair_from_book(task="multitask", path=path)
    elif request_item.task == "multiple-choices":
        qg_result = model.generate_qa_pair_from_book(task="mc", path=path)
    else:
        return HTTPException
    elapsed_time = datetime.now() - start_time
    elapsed_time = elapsed_time.total_seconds() * 1000
    for i, item in enumerate(qg_result):
        qg_result[i] = item
    result = transform_output(
        qg_result, task=request_item.task, domain="history", elapsed_time=elapsed_time
    )

    log_item = copy.deepcopy(result)
    generate_mongodb.insert_one(log_item)
    logger.info(f"Ready to return: {result}")
    torch.cuda.empty_cache()
    gc.collect()
    return result


@router.post("/v1/history/generate_from_file")
async def upload_file(task: str = "", file: UploadFile = File(...)):
    raw_context_list = file_preprocess(file)
    preprocessor = Preprocessor("configs/parse_pdf_config.yml")
    context_list = preprocessor.preprocess_pdf_context(raw_context_list)
    context_text = split_context("\n".join(context_list))
    try:
        result = generate(context_text=context_text, model=model, task=task, lang="vi")
        log_item = copy.deepcopy(result)
        generate_mongodb.insert_one(log_item)
        return result
    except Exception as e:
        logger.exception(e)
        return None
