import json
from collections import defaultdict
from typing import *
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass
from glob import glob
import os
from datetime import datetime


class HardVoting:
    """
    result_json_path_list인자에 제이슨파일path를 리스트로 담아서 넣어주믄됨..
    """

    def __init__(self, result_json_path_list: List[str]):
        """
        result_json_path_list인자에 제이슨파일path를 리스트로 담아서 넣어주믄됨..
        """
        self.result_json_path_list: List[str] = result_json_path_list
        self.result_list: List[List[str]] = list(
            map(self.__callback_func, self.result_json_path_list)
        )
        with open(self.result_json_path_list[0], "r") as f:
            self.ids = list(json.load(f).keys())

        return None

    def __callback_func(self, path: str):
        with open(path, "r") as f:
            result_list = json.load(f)

        return list(result_list.values())

    def __vote(self, infer_word_list: List[str]) -> str:
        answer_dict = defaultdict(int)
        for word in infer_word_list:
            answer_dict[word] += 1
        answer_dict = list(answer_dict.items())
        answer_dict = sorted(answer_dict, key=lambda x: -x[1])
        if len(answer_dict) > 1:
            for i in range(len(answer_dict)):
                if answer_dict[i][0] == "[CLS]":
                    answer_dict.pop(i)
                    break

        if len(answer_dict) > 1 and len(answer_dict[0][0]) > 20:
            answer_dict.pop(0)

        return answer_dict[0][0]

    def voting(self) -> dict:
        self.voted_answer_dict = {}
        for i, infer_word_list in enumerate(zip(*self.result_list)):
            # if i < 5:
            #     print(infer_word_list)
            assert len(infer_word_list) == len(self.result_list)

            voted_answer = self.__vote(list(infer_word_list))
            self.voted_answer_dict[self.ids[i]] = voted_answer

        return self.voted_answer_dict


if __name__ == "__main__":

    @dataclass(frozen=True)
    class Args:
        path: str = "./test_results"
        output_path: str = "./emsemble_results"

    parser = HfArgumentParser([Args])
    (args,) = parser.parse_args_into_dataclasses()
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    result_json_path_list: List[str] = glob(os.path.join(args.path, "*.json"))

    hard_voting = HardVoting(result_json_path_list)
    final_answer = hard_voting.voting()
    with open(
        os.path.join(args.output_path, f"emsemble_{datetime.now().__str__()}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_answer, f, ensure_ascii=False)
