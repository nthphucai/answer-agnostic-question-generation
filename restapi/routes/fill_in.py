import os
import random
import sys
import time

import fastapi
import torch
from fastapi.openapi.docs import get_swagger_ui_html

from restapi.schema.context_answer import FIBRequestItem, FIBWithGivenWordItem


sys.path.append("utils")

from fib_questgen.models.qa_generator import QAGenerator
from mongodb.mongo_client import connect_mongodb
from questgen.utils.file_utils import read_yaml_file
from restapi.utils import shuffle_answers


router = fastapi.APIRouter()

PIPELINE_CONFIG_PATH = os.getenv(
    "PIPELINE_CONFIG_PATH", "configs/faqg_pipeline_t5_vi_base_hl.yaml"
)
random.seed(1358)
torch.manual_seed(1358)
# Read the configuration for question generation pipeline
config = read_yaml_file(PIPELINE_CONFIG_PATH)

model = QAGenerator()

generate_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "generate_collection")
feedback_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "feedback_collection")


@router.get("/v1/english/fillin/healthcheck")
async def healthcheck():
    return {"message": "OK"}


@router.get("/v1/english/fillin/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url="/v1/english/fillin/openapi.json",
        title="API",
    )


@router.post("/v1/english/fillin")
async def generate_fib(request_item: FIBRequestItem):
    start_time = time.time()
    context = request_item.context
    qa_pairs = model.generate_paragraph(
        context=context, num_blank=request_item.num_blank
    )
    for i, item in enumerate(qa_pairs["results"]):
        new_item = {"question": item["question"]}
        new_item.update(shuffle_answers(item["answers"]))
        qa_pairs["results"][i] = new_item
    qa_pairs["time"] = time.time() - start_time
    return qa_pairs


@router.post("/v1/english/fillin_with_given_word")
async def generate_fib(request_item: FIBWithGivenWordItem):
    start_time = time.time()
    context = request_item.context
    word = request_item.word
    if word == "None":
        word = None
    qa_pairs = model.generate_sentence(context=context, word=word)
    for i, item in enumerate(qa_pairs["results"]):
        new_item = {"question": item["question"]}
        new_item.update(shuffle_answers(item["answers"]))
        qa_pairs["results"][i] = new_item
    qa_pairs["time"] = time.time() - start_time
    return qa_pairs
