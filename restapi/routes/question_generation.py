import copy
import gc
import os
import random
import sys
import time

from fastapi import HTTPException


sys.path.append("utils")

from datetime import datetime

import fastapi
import torch
from loguru import logger

from fib_questgen.models.qa_generator import QAGenerator
from mongodb.mongo_client import connect_mongodb
from questgen.inference import Inference
from questgen.utils.file_utils import read_yaml_file
from restapi.schema.context_answer import (
    BookRequestItem,
    FeedBack,
    FIBRequestItem,
    QuestgenRequestItem,
)


router = fastapi.APIRouter()

PIPELINE_CONFIG_PATH = os.getenv(
    "PIPELINE_CONFIG_PATH", "configs/faqg_pipeline_t5_vi_base_hl.yaml"
)
random.seed(1358)
torch.manual_seed(1358)
# Read the configuration for question generation pipeline
config = read_yaml_file(PIPELINE_CONFIG_PATH)

english_model = Inference(
    config_aqg_path=PIPELINE_CONFIG_PATH,
    multitask_model_name_or_path="/home/hungtx/questgen/output/models/english/squad_simple_85k",
    mc_model_name_or_path="/home/hungtx/questgen/output/models/mc_english",
    tokenizer_name_or_path="/home/hungtx/questgen/output/models/english/squad_simple_85k",
)

history_model = Inference(
    config_aqg_path=PIPELINE_CONFIG_PATH,
    multitask_model_name_or_path="/home/phucnth/aqg/output/mix-finetuning/phase-2/full-data/v1.2",
    mc_model_name_or_path="/home/phucnth/aqg/output/models/mc/transformers_v4.5.1/v1.2",
    tokenizer_name_or_path="/home/hungtx/questgen/output/models/history/full-data",
    use_summary=True,
)

faq_model = Inference(
    config_aqg_path=PIPELINE_CONFIG_PATH,
    multitask_model_name_or_path="/home/phucnth/aqg/output/models/simple-question/viquad/transformers_v.4.5.1/v1.4",
    tokenizer_name_or_path="/home/phucnth/aqg/output/models/simple-question/viquad/transformers_v.4.5.1/v1.4",
    use_summary=True,
)

model_map = {
    "english": english_model,
    "history": history_model,
    "faq": faq_model,
}

fib_model = QAGenerator()

generate_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "generate_collection")
feedback_mongodb = connect_mongodb(PIPELINE_CONFIG_PATH, "feedback_collection")


def transform_output(qg_result, task, domain, elapsed_time):
    result = [{"context": "", "result": []}]
    for item in qg_result:
        if task == "simple-question":
            qg_item = {
                "question": item["question"],
                "answer": item["answer"],
                "key_info": item["key_info"],
            }
        elif task == "multiple-choices":
            qg_item = {
                "question": item["question"][0],
            }
            qg_item.update(shuffle_answers(item["answers"]))
            qg_item["key_info"] = item["key_info"]
        else:
            raise Exception("task arg must in ['simple-question', 'multiple-choices']")
        if item["context"] != result[-1]["context"]:
            result.append({"context": item["context"], "results": [qg_item]})
        else:
            result[-1]["results"].append(qg_item)
    del result[0]
    return {"task": task, "domain": domain, "data": result, "time": elapsed_time}


def shuffle_answers(answers):
    answers_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    option_index = list(range(len(answers)))
    random.shuffle(option_index)
    options = [
        answers_map[i] + ". " + answers[index] for i, index in enumerate(option_index)
    ]
    answer = answers_map[option_index.index(0)]
    return {"options": options, "answer": answer}


def expand_context(sentence_list):
    context = ""
    for i, sent in enumerate(sentence_list):
        if len(context) > 1300:
            return context, sentence_list[i - 1 :]
        context += f".{sent}"


def split_context(context):
    sentence_list = context.split(".")
    sub_context_list = []
    while len(context) > 1300:
        sub_context, sentence_list = expand_context(sentence_list)
        sub_context_list.append(sub_context)
        context = ".".join(sentence_list)
    sub_context_list.append(context)
    return sub_context_list


@router.post("/api/v1/generate")
async def generate(request_item: QuestgenRequestItem):
    try:
        start_time = datetime.now()
        logger.info("Generating questions from a context")
        model = model_map[request_item.domain]
        context_text = request_item.context
        context_text = split_context(context_text)
        lang = "eng" if request_item.domain == "english" else "vi"
        if request_item.task == "simple-question":
            qg_result = model.create(task="multitask", context=context_text)
        elif request_item.task == "multiple-choices":
            qg_result = model.create(task="mc", context=context_text, lang=lang)
        else:
            return HTTPException
        elapsed_time = datetime.now() - start_time
        elapsed_time = elapsed_time.total_seconds() * 1000
        result = transform_output(
            qg_result, request_item.task, request_item.domain, elapsed_time
        )
        log_item = copy.deepcopy(result)
        generate_mongodb.insert_one(log_item)
        logger.info(f"Ready to return: {result}")
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        logger.exception(e)
        return None
    return result


@router.post("/api/v1/generate_from_book")
async def generate_from_book(request_item: BookRequestItem):
    # try:
    logger.info("Generating questions from a context")
    model = model_map["history"]
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


@router.post("/api/v1/generate_fib")
async def generate_fib(request_item: FIBRequestItem):
    start_time = time.time()
    context = request_item.context
    qa_pairs = fib_model.generate_paragraph(
        context=context, num_blank=request_item.num_blank
    )
    for i, item in enumerate(qa_pairs["results"]):
        new_item = {"question": item["question"]}
        new_item.update(shuffle_answers(item["answers"]))
        qa_pairs["results"][i] = new_item
    qa_pairs["time"] = time.time() - start_time
    return qa_pairs


@router.post("/api/v1/feedback", include_in_schema=False, status_code=200)
async def feedback(context: FeedBack):
    try:
        result = {
            "task": context.task,
            "domain": context.domain,
            "results": context.results,
            "time": context.time,
            "location_time": str(datetime.now()),
        }
        feedback_mongodb.insert_one(result)
        return {"message": "feedback success"}

    except Exception as e:
        logger.exception(e)
        return {"error": "validation error"}
