import gc
import math
import random
from datetime import datetime
from http.client import HTTPException
from io import BytesIO
from typing import List

import requests
import torch
from fastapi import UploadFile
from loguru import logger
from tqdm import trange

import docx
import fitz
from questgen.inference import Inference


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


def file_preprocess(file: UploadFile):
    file_type = file.filename.split(".")[-1]
    if file_type == "docx":
        doc = docx.Document(BytesIO(file.file.read()))
        context_list = []
        for para in doc.paragraphs:
            context_list.append(para.text)
    elif file_type == "pdf":
        NUM_PAGES = 6
        input_pdf = fitz.open(stream=BytesIO(file.file.read()), filetype="pdf")
        context_list = []
        for i in trange(
            math.ceil(len(input_pdf) / NUM_PAGES), desc="Splitting pdf file: "
        ):
            output_pdf = fitz.open()
            if i == math.ceil(len(input_pdf) / NUM_PAGES):
                max_pages = len(input_pdf)
            else:
                max_pages = (i + 1) * NUM_PAGES - 1
            output_pdf.insert_pdf(input_pdf, from_page=i * NUM_PAGES, to_page=max_pages)
            output = requests.post(
                "https://fscidemo.fschool.dev.ftech.ai/doc/inference/file?responseFormat=raw",
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Accept-Encoding": "gzip, deflate, br",
                },
                files=[
                    ("files", ("context.pdf", output_pdf.tobytes(), "application/pdf"))
                ],
                data={},
                timeout=5,
            )
            if output.status_code == 200:
                context_list.extend(output.json()["questions"])
            else:
                logger.error("Something wrong with ocr parsing server")

    else:
        return {"error": f"file type {file_type} is not supported"}

    return context_list


def generate(context_text, model: Inference, task: str, lang: str):
    start_time = datetime.now()
    context_batched = batching(context_text, batch_size=16)
    qg_result = []
    if task == "simple-question":
        for batch in context_batched:
            print(len(batch))
            qg_out = model.create(task="multitask", context=batch)
            qg_result.extend(qg_out)
    elif task == "multiple-choices":
        for batch in context_batched:
            print(len(batch))
            qg_out = model.create(task="mc", context=batch, lang=lang)
            qg_result.extend(qg_out)
    else:
        return HTTPException
    elapsed_time = datetime.now() - start_time
    elapsed_time = elapsed_time.total_seconds() * 1000
    result = transform_output(qg_result, task, "english", elapsed_time)
    torch.cuda.empty_cache()
    gc.collect()
    return result


def batching(text_list: List[str], batch_size: int = 16) -> List[List[str]]:
    """
    Split a text list in to batch
    Args:
        text_list: (`List[str]`) Text list need to be batched
        batch_size: (`int`) Batch size
    Returns: (`List`) List of minibatch
    """
    result_list = []
    data_len = len(text_list)
    for idx in range(int(data_len / batch_size) + 1):
        if (idx + 1) * batch_size >= data_len:
            end_offset = data_len
        else:
            end_offset = (idx + 1) * batch_size
        result_list.append(text_list[idx * batch_size : end_offset])
    if not result_list[-1]:
        return result_list[:-1]
    return result_list
