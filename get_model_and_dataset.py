from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BatchEncoding
import torch
from datasets import load_from_disk
from toolkits import prepare_train_features, QADataset
from transformers.hf_argparser import HfArgumentParser
import random
from typing import *
import numpy as np
from dataclasses import dataclass
import requests
import json
import os


if __name__ == "__main__":

    @dataclass(frozen=True)
    class Args:
        name_or_path: str = "klue/roberta-large"
        train_dataset_path: str = "./data/train_dataset"
        test_dataset_path: str = "./data/test_dataset"
        output_path = "./bin"

    parser = HfArgumentParser([Args])
    (args,) = parser.parse_args_into_dataclasses()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    model = AutoModelForQuestionAnswering.from_pretrained(args.name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)

    torch.save(model, os.path.join(args.output_path, "model.pt"))
    torch.save(tokenizer, os.path.join(args.output_path, "tokenizer.pt"))

    train_dataset = load_from_disk(args.train_dataset_path)

    train_qa_dataset = QADataset(
        tokenizer,
        train_dataset["train"]["question"],
        train_dataset["train"]["context"],
        list(map(lambda x: x["text"][0], train_dataset["train"]["answers"])),
        list(map(lambda x: x["answer_start"][0], train_dataset["train"]["answers"])),
    )

    torch.save(train_qa_dataset, os.path.join(args.output_path, "train_dataset.pt"))

    valid_qa_dataset = QADataset(
        tokenizer,
        train_dataset["validation"]["question"],
        train_dataset["validation"]["context"],
        list(map(lambda x: x["text"][0], train_dataset["validation"]["answers"])),
        list(
            map(lambda x: x["answer_start"][0], train_dataset["validation"]["answers"])
        ),
    )

    torch.save(valid_qa_dataset, os.path.join(args.output_path, "valid_dataset.pt"))

    test_dataset = load_from_disk(args.test_dataset_path)
    context_list = []
    for data in test_dataset["validation"]:
        query = data["question"]

        res = requests.get(url=f"http://49.164.19.151:9200/?q={query}")
        context_list.append(res.json())

    with open(
        os.path.join(args.output_path, "retrieved_context_list.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(context_list, f, ensure_ascii=False)
    print("🐟완료🤗")
