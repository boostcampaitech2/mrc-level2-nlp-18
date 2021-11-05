import torch
import logging
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, BatchEncoding
import torch
from datasets import load_from_disk
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import random
from typing import *
import numpy as np
from dataclasses import dataclass
import pandas as pd


class ValidQAModel:
    def __init__(
        self,
        model,
        tokenizer,
        question_list: Iterable[str],
        contexts_list: Iterable[Iterable[str]],
        no_token_type_ids: bool = False,
        cls_token_id: int = 0,
    ) -> List[str]:
        """
        args \n
        `question_list :Iterable[str]` \n
        질의어들이 담긴 리스트 \n
        `contexts_list :Iterable[Iterable[str]]` \n
        해당 질의어와 연관된 context들이 담긴 리스트, 한 질의어당 여러가지(top_k 만큼의) context들이 담길 것이다. 이로 인해 `Iterable[Iterable[str]]`구조가 된다.
        `no_token_type_ids :bool` \n
        roberta같은 모델들은 .forward메서드의 파라미터로  token_type_ids인자가 없다. 그런 모델들에 token_type_ids를 전달하면 에러가 발생한다. (ex klue-roberta) 따라서 그런 모델들을 사용할 때는 해당 옵션을 True로 설정해야 한다. \n
        `cls_token_id :int` \n
        토크나이저에서 CLS토큰의 ID를 입력하면 된다. 보통은 0이 CLS 토큰 아이디라서 디폴트를 0으로 해놨는데 가끔씩 아닌 모델들이 있다. 그런 모델들은 수동으로 조정해주면 된다.
        """
        (  # set a members of Class
            self.question_list,
            self.model,
            self.tokenizer,
            self.contexts_list,
            self.no_token_type_ids,
            self.cls_token_id,
        ) = (
            question_list,
            model,
            tokenizer,
            contexts_list,
            no_token_type_ids,
            cls_token_id,
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.final_pred_list: List[str] = []
        self.type_conversion()
        self.stability_test()

    def stability_test(self):
        assert (
            self.model.name_or_path == self.tokenizer.name_or_path
        ), "토크나이저와 모델이 서로 호환되지 않는 것 같습니다."
        assert (
            type(self.contexts_list) == list or type(self.contexts_list) == np.ndarray
        )
        assert (
            type(self.contexts_list[0]) == list
            or type(self.contexts_list[0]) == np.ndarray
        )
        assert type(self.contexts_list[0][0]) == str
        assert (
            type(self.question_list) == list or type(self.question_list) == np.ndarray
        )

        assert type(self.question_list[0]) == str

        for property in dir(self):
            assert property in [
                "_ValidQAModel__infer_st_and_end",
                "__call__",
                "__class__",
                "__delattr__",
                "__dict__",
                "__dir__",
                "__doc__",
                "__eq__",
                "__format__",
                "__ge__",
                "__getattribute__",
                "__gt__",
                "__hash__",
                "__init__",
                "__init_subclass__",
                "__le__",
                "__lt__",
                "__module__",
                "__ne__",
                "__new__",
                "__reduce__",
                "__reduce_ex__",
                "__repr__",
                "__setattr__",
                "__sizeof__",
                "__str__",
                "__subclasshook__",
                "__weakref__",
                "contexts_list",
                "final_pred_list",
                "model",
                "no_token_type_ids",
                "question_list",
                "softmax",
                "tokenizer",
                "cls_token_id",
                "type_conversion",
                "stability_test",
            ], f"{property}는 존재하면 안 되는 메서드 혹은 어트리뷰트입니다. 🌊"
        print("👩모든 test case를 통과했습니다.")

    def type_conversion(self):
        if type(self.contexts_list) == pd.Series:
            self.contexts_list = self.contexts_list.values
        if type(self.question_list) == pd.Series:
            self.question_list = self.question_list.values

    def __call__(self):
        for q_idx, question in enumerate(self.question_list):
            context_list_about_question: List[str] = self.contexts_list[q_idx]
            TOP_K: int = len(context_list_about_question)
            assert len([question] * TOP_K) == len(context_list_about_question)
            assert type(question) == str

            # 어떤 한 퀘스전(:str)에 대해 검색된 복수의 관련 문서들을 엮어
            # 어떤 한 퀘스전의 밸리드 데이터를 만든다.
            valid_data = QADataset(
                tokenizer=self.tokenizer,
                question=[question] * TOP_K,
                context=context_list_about_question,
            )  # valid_Data를 반환 받음
            input_id_list: np.ndarray = (
                valid_data[:]["input_ids"].detach().cpu().numpy()
            )
            st_max_odds, st_max_labels, end_max_odds, end_max_labels = [], [], [], []

            for i in range(len(valid_data)):
                (
                    st_max_odd,
                    st_max_label,
                    end_max_odd,
                    end_max_label,
                ) = self.__infer_st_and_end(valid_data[i])
                st_max_odds.append(st_max_odd)
                st_max_labels.append(st_max_label)
                end_max_labels.append(end_max_label)
                end_max_odds.append(end_max_odd)

            pred_list = []
            for idx, (
                st_max_odd,
                st_max_label,
                end_max_odd,
                end_max_label,
            ) in enumerate(
                zip(st_max_odds, st_max_labels, end_max_odds, end_max_labels)
            ):
                if (
                    st_max_label == self.cls_token_id
                    or end_max_label == self.cls_token_id
                ):
                    continue  # 😈 스타트 포지션 혹은 엔드 포지션을 CLS 토큰이라도 추정했다면 버리라는 얘기다.
                pred_list.append(
                    (
                        st_max_label,
                        end_max_label,
                        (st_max_odd + end_max_odd),
                        idx,
                    )
                )
            try:
                if pred_list == []:
                    pred_list = [(0, 0, 0, 0)]
                pred_start, pred_end, pred_odd, pred_idx = sorted(
                    pred_list, key=lambda x: -x[2]
                )[0]

            except:
                print(
                    "🤦‍♀️🤦‍♂️지금 이 메시지가 보인다면 코드에 문제가 있다는 이야기입니다.",
                    pred_list,
                    question,
                    st_max_odds,
                    st_max_labels,
                    end_max_odds,
                    end_max_labels,
                )

            pred_token: List[str] = input_id_list[pred_idx][pred_start : pred_end + 1]
            pred_word: str = self.tokenizer.decode(pred_token).replace("#", "")
            self.final_pred_list.append(pred_word)

        return self.final_pred_list

    def __infer_st_and_end(self, data):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(device)

        with torch.no_grad():
            output = self.model(
                input_ids=data["input_ids"].unsqueeze(dim=0).to(device),
                token_type_ids=data["token_type_ids"].unsqueeze(dim=0).to(device)
                if self.no_token_type_ids == False
                else None,
                attention_mask=data["attention_mask"].unsqueeze(dim=0).to(device),
            )

            torch.cuda.empty_cache()

            st_max_odd: float = self.softmax(output.start_logits).max(dim=-1)[0].item()
            st_max_label: int = self.softmax(output.start_logits).max(dim=-1)[1].item()
            end_max_odd: float = self.softmax(output.end_logits).max(dim=-1)[0].item()
            end_max_label: int = self.softmax(output.end_logits).max(dim=-1)[1].item()

            del output  # 메모리 정리
            torch.cuda.empty_cache()

            return st_max_odd, st_max_label, end_max_odd, end_max_label


class TrainQAModel:
    def __init__(
        self,
        model,
        train_dataset,
        valid_dataset,
        output_path: str,
        lr: float = 0.00005,
        batch_size: int = 16,
        no_token_type_ids=False,
        OptimizerClass=torch.optim.Adam,
        wandb_run=None,
        logging_freq=10,
        warmup_steps: int = 0,
        epoch: int = 10,
    ):

        (
            self.model,
            self.no_token_type_ids,
            self.output_path,
            self.train_dataset,
            self.valid_dataset,
            self.batch_size,
            self.lr,
            self.wandb_run,
            self.__logging_freq,
            self.epoch,
        ) = (
            model,
            no_token_type_ids,
            output_path,
            train_dataset,
            valid_dataset,
            batch_size,
            lr,
            wandb_run,
            logging_freq,
            epoch,
        )

        self.loss = None
        self.train_acc = None
        self.valid_acc = None
        self.__logger = logging.getLogger()

        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)

        self.__logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler = logging.FileHandler(os.path.join(output_path, "log.log"))
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(stream_handler)
        self.optim = OptimizerClass(self.model.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optim,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.epoch
            * (len(self.train_dataset) // self.batch_size),
        )
        self.__call__ = self.train

    def test_safety(self):
        for key in list(dir(self)):

            assert key in [
                "_TrainQAModel__k_fold_ratio",
                "_TrainQAModel__logger",
                "_TrainQAModel__logging_freq",
                "epoch",
                "__call__",
                "__class__",
                "__delattr__",
                "__dict__",
                "__dir__",
                "__doc__",
                "__eq__",
                "__format__",
                "__ge__",
                "__getattribute__",
                "__gt__",
                "__hash__",
                "__init__",
                "__init_subclass__",
                "__le__",
                "__lt__",
                "__module__",
                "__ne__",
                "__new__",
                "__reduce__",
                "__reduce_ex__",
                "__repr__",
                "__setattr__",
                "__sizeof__",
                "__str__",
                "__subclasshook__",
                "__weakref__",
                "batch_size",
                "dataset",
                "train_dataset",
                "valid_dataset",
                "device",
                "wandb_run",
                "loss",
                "scheduler",
                "lr",
                "k_fold",
                "model",
                "no_token_type_ids",
                "optim",
                "output_path",
                "test_safety",
                "train",
                "train_acc",
                "valid_acc",
                "save_cp",
                "extract_best_model",
                "train",
                "valid",
                "scheduler",
            ], f"""🤦‍♀️ {key}는 인스턴스 안에 존재할 수 없는 attribute( or method)입니다.
            Instancede의 Initialization이 잘못된 것 같습니다."""
        print("👩 test case 통과!")

    def train(self):
        train_batches = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.model = self.model.to(self.device)
        self.model.train()
        self.__logger.info(f"👩학습을 시작합니다.")
        st_pred = []
        end_pred = []
        st_true = []
        end_true = []

        for i, batch in enumerate(train_batches):
            st_true += list(batch["start_positions"].cpu().numpy())
            end_true += list(batch["end_positions"].cpu().numpy())

            self.optim.zero_grad()
            output = self.model(
                input_ids=batch["input_ids"].to(self.device),
                token_type_ids=None
                if self.no_token_type_ids
                else batch["token_type_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                start_positions=batch["start_positions"].to(self.device),
                end_positions=batch["end_positions"].to(self.device),
            )

            st_pred += list(output.start_logits.argmax(dim=-1).cpu().numpy())
            end_pred += list(output.end_logits.argmax(dim=-1).cpu().numpy())
            self.loss = output.loss.item()
            output.loss.backward()
            self.optim.step()
            self.scheduler.step()
            if i % self.__logging_freq == self.__logging_freq - 1:
                self.train_acc = accuracy_score(st_true + end_true, st_pred + end_pred)
                self.__logger.info(f"🔨train acc : {self.train_acc}")
                if self.wandb_run is not None:
                    self.wandb_run.log({"train_acc": self.train_acc})
        self.__logger.info("👩학습을 종료합니다 :D")

    def valid(self):
        valid_batches = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.model = self.model.to(self.device)
        st_pred = []
        end_pred = []
        st_true = []
        end_true = []
        with torch.no_grad():
            self.__logger.info(f"👩검증을 시작합니다.")
            self.model.eval()
            for i, batch in enumerate(valid_batches):
                st_true += list(batch["start_positions"].cpu().numpy())
                end_true += list(batch["end_positions"].cpu().numpy())

                output = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    token_type_ids=None
                    if self.no_token_type_ids
                    else batch["token_type_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    start_positions=batch["start_positions"].to(self.device),
                    end_positions=batch["end_positions"].to(self.device),
                )

                st_pred += list(output.start_logits.argmax(dim=-1).cpu().numpy())
                end_pred += list(output.end_logits.argmax(dim=-1).cpu().numpy())

                if i % self.__logging_freq == self.__logging_freq - 1:
                    self.valid_acc = accuracy_score(
                        st_true + end_true, st_pred + end_pred
                    )
                    self.__logger.info(f"🔨valid acc : {self.valid_acc}")

        if self.wandb_run is not None:
            self.wandb_run.log({"valid_acc": self.valid_acc})

    def save_cp(self):
        cp_path = os.path.join(self.output_path, "cp_list")
        if not os.path.isdir(cp_path):
            os.makedirs(cp_path)
        torch.save(
            self.model,
            os.path.join(cp_path, f"{self.valid_acc}.pt"),
        )
        print("모델 저장 완료!")

    def extract_best_model(self):
        cp_list = os.listdir(os.path.join(self.output_path, "cp_list"))
        cp_list.sort(reverse=True, key=lambda x: eval(x.split(".pt")[0]))
        cp_list.pop(0)
        for cp_file in cp_list:
            os.remove(os.path.join(self.output_path, "cp_list", cp_file))


def prepare_train_features(
    tokenizer,
    question: Iterable[str],
    context: Iterable[str],
    answer_text: Iterable[str] = None,
    answer_start: Iterable[int] = None,
    max_seq_length: int = 384,
    doc_stride: int = 128,
    verbose=False,
) -> BatchEncoding:

    # 타입 체킹 과정
    assert type(question) in [list, pd.Series, np.ndarray]
    assert type(context) in [list, pd.Series, np.ndarray]
    if not answer_text is None:
        assert type(answer_text) in [list, pd.Series, np.ndarray]
    if not answer_start is None:
        assert type(answer_start) in [list, pd.Series, np.ndarray]

    if type(question) == pd.Series:
        question = question.values
    if type(context) == pd.Series:
        context = context.values
    if not answer_text is None:
        if type(answer_text) == pd.Series:
            answer_text = answer_text.values
    if not answer_start is None:
        if type(answer_start) == pd.Series:
            answer_start = answer_start.values

    if type(question) == np.ndarray:
        question = list(question)
    if type(context) == np.ndarray:
        context = list(context)
    if not answer_text is None:
        if type(answer_text) == np.ndarray:
            answer_text = list(answer_text)
    if not answer_start is None:
        if type(answer_start) == np.ndarray:
            answer_start = list(answer_start)

    # 주어진 텍스트를 토크나이징함
    # 이 때 텍스트의 길이가 max_seq_length를 넘으면 stride만큼 슬라이딩하며 여러 개로 나눔
    # 즉, 하나의 example에서 일부분이 겹치는 여러 sequence(feature)가 생길 수 있음
    tokenized_examples = tokenizer(
        question,
        context,
        truncation="only_second",  # max_seq_length까지 truncate함 / pair의 두번째 파트(context)만 잘라냄
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,  # 길이를 넘어가는 토큰들을 반환할 것인지
        return_offsets_mapping=True,  # 각 토큰에 대해 (char_start, char_end) 정보를 반환한 것인지
        padding="max_length",
    )
    if answer_text and answer_start:
        # example 하나가 여러 sequence에 대응하는 경우를 위해 매핑이 필요
        overflow_to_sample_mapping = tokenized_examples.pop(
            "overflow_to_sample_mapping"
        )
        # offset_mappings으로 토큰이 원본 context 내 몇번째 글자부터 몇번째 글자까지 해당하는지 알 수 있음
        offset_mapping: Iterable[Tuple[int, int]] = tokenized_examples.pop(
            "offset_mapping"
        )
        # print('gfg', overflow_to_sample_mapping)
        # 정답지를 만들기 위한 리스트
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # print(offsets)
            input_ids = tokenized_examples["input_ids"][i]
            # offsets, input_ids는 일단적은 BatchEncoding의 어트리뷰트와 동일.

            cls_index: int = input_ids.index(tokenizer.cls_token_id)

            # 해당 example에 해당하는 sequence를 찾음
            sequence_ids = tokenized_examples.sequence_ids(i)

            # sequence가 속하는 example을 찾는다
            example_index = overflow_to_sample_mapping[i]
            # answers :Dict[
            #   Literal["answer_start", "text"]: Union[int, str]
            # ] = answer[example_index]# answer의 원본에 해당된다.

            # 텍스트에서 answer의 시작점, 끝점
            answer_start_offset: int = answer_start[
                example_index
            ]  # answers["answer_start"][0]
            answer_end_offset: int = answer_start_offset + len(
                answer_text[example_index]
            )

            # 텍스트에서 현재 span의 시작 토큰 인덱스
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:  # 1이 SEP토큰인가? ## ㅇㅇ 그렇네
                token_start_index += 1

            # 텍스트에서 현재 span 끝 토큰 인덱스
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # answer가 현재 span을 벗어났는지 체크
            if not (
                offsets[token_start_index][0] <= answer_start_offset
                and offsets[token_end_index][1] >= answer_end_offset
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index와 token_end_index를 answer의 시작점과 끝점으로 옮김
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= answer_start_offset
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= answer_end_offset:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
        if verbose:
            for _ in range(10):
                i = random.randint(0, len(tokenized_examples["input_ids"]) - 1)
                input_ids: str = tokenized_examples["input_ids"][i]
                st: int = tokenized_examples["start_positions"][i]
                end: int = tokenized_examples["end_positions"][i]
                infer = tokenizer.decode(input_ids[st : end + 1])
                sentence = tokenizer.decode(input_ids[:st])
                print(i)
                print("👩", sentence)
                print("💻", infer, "\n")

    return tokenized_examples


@dataclass(frozen=True)
class QADatasetArgs:
    tokenizer: Callable
    question: Iterable[str]
    context: Iterable[str]
    answer_text: Iterable[str]
    answer_start: Iterable[int]
    max_seq_length: int
    doc_stride: int


class QADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        question: Iterable[str],
        context: Iterable[str],
        answer_text: Iterable[str] = None,
        answer_start: Iterable[int] = None,
        max_seq_length: int = 384,
        doc_stride: int = 128,
    ):
        self.args = QADatasetArgs(
            tokenizer=tokenizer,
            question=np.array(question) if type(question) == list else question,
            context=np.array(context) if type(context) == list else context,
            answer_text=np.array(answer_text)
            if type(answer_text) == list
            else answer_text,
            answer_start=np.array(answer_start)
            if type(answer_start) == list
            else answer_start,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
        )

        self.batch_encoding: BatchEncoding = prepare_train_features(
            tokenizer=tokenizer,
            question=question,
            context=context,
            answer_text=answer_text,
            answer_start=answer_start,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
        )

    def __getitem__(self, idx):
        return {
            key: torch.tensor(value[idx]) for key, value in self.batch_encoding.items()
        }

    def __len__(self):
        return self.batch_encoding["input_ids"].__len__()
