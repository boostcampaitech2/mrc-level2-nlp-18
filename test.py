from pandas.io import parsers
from transformers.hf_argparser import HfArgumentParser
from toolkits import ValidQAModel
import torch
from datasets import load_from_disk
import json
from dataclasses import dataclass, field
from datetime import datetime
from glob import glob
import os

if __name__ == "__main__":

    @dataclass
    class TestArgument:
        model_cp_path: str = glob("./train_results/cp_list/*.pt")[0]
        context_list_path: str = "./bin/retrieved_context_list.json"
        test_dataset_path: str = "./data/test_dataset"
        tokenizer_cp_path: str = "./bin/tokenizer.pt"
        output_path: str = "./test_results"
        no_token_type_ids: bool = True

    parser = HfArgumentParser([TestArgument])
    (test_args,) = parser.parse_args_into_dataclasses()
    if not os.path.isdir(test_args.output_path):
        os.makedirs(test_args.output_path)

    with open(test_args.context_list_path, "r", encoding="utf-8") as f:
        context_list = json.load(f)

    testset = load_from_disk(test_args.test_dataset_path)
    model = torch.load(test_args.model_cp_path)
    tokenizer = torch.load(test_args.tokenizer_cp_path)
    question_list: list = testset["validation"]["question"]
    id_list: list = testset["validation"]["id"]

    valid_model = ValidQAModel(
        model=model,
        tokenizer=tokenizer,
        question_list=question_list,
        contexts_list=context_list,
        no_token_type_ids=test_args.no_token_type_ids,
    )

    result: list = valid_model()

    answer: dict = {}
    for id, word in zip(id_list, result):
        answer[id] = word

    with open(
        os.path.join(test_args.output_path, f"{datetime.now().__str__()}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(answer, f, ensure_ascii=False)
