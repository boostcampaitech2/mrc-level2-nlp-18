import os
import json
import time
import faiss
import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union

from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
  def forward(self, input_ids, 
              attention_mask=None, token_type_ids=None): 
  
        outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)
      
        pooled_output = outputs[1] # cls 토큰

        return pooled_output

class DenseRetrieval:
    def __init__(self,
        args,
        dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder
     ):
        """
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        """

        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.prepare_in_batch_negative(num_neg=num_neg)


        with open(os.path.join("../data", "wikipedia_documents.json"), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        
        self.texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  # set 은 매번 순서가 바뀌므로
        # print(f"Lengths of unique contexts : {len(texts)}")
        # ids = list(range(len(texts)))
        
        passage_seqs = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inf_dataset = TensorDataset(
            passage_seqs["input_ids"],
            passage_seqs["attention_mask"],
            passage_seqs["token_type_ids"]
        )
        self.inf_dataloader = DataLoader(
            inf_dataset,
            batch_size=self.args.per_device_train_batch_size,
            drop_last = True
        )
    
    # p_embs
    def get_p_embs():
        with torch.no_grad():
            self.p_encoder.eval()

            self.p_embs = []
            for batch in self.inf_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                }
                p_emb = self.p_encoder(**p_inputs)
                self.p_embs.append(p_emb)

        # (num_passage, emb_dim)
        self.p_embs = torch.stack(
            self.p_embs, dim=0
        ).view(len(self.p_embs)*self.args.per_device_train_batch_size, -1)

        return self.p_embs

    def prepare_in_batch_negative(self,
        dataset=None,
        num_neg=2,
        tokenizer=None
        ):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.
        corpus = np.array(list(set([example for example in dataset["context"]]))) # 중복 제거
        p_with_neg = []

        for c in dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg) # 2개 랜덤으로 뽑기

                if not c in corpus[neg_idxs]: # 랜덤으로 뽑은 passage가 c와 다르면 선택
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c) # 현재 passge 
                    p_with_neg.extend(p_neg) # negative sample 2개
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg+1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg+1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"], 
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size
        )

        valid_seqs = tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        passage_dataset = TensorDataset(
            valid_seqs["input_ids"],
            valid_seqs["attention_mask"],
            valid_seqs["token_type_ids"]
        )
        self.passage_dataloader = DataLoader(
            passage_dataset,
            batch_size=self.args.per_device_train_batch_size
        )

    def train(self, args=None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()

        self.p_encoder.train()
        self.q_encoder.train()

        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):

        for _ in train_iterator:
            with tqdm(self.train_dataloader, unit="batch") as tepoch:

                for batch in tepoch:
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        "token_type_ids": batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    # (batch_size*(num_neg+1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size*, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = torch.transpose(p_outputs.view(batch_size,self.num_neg+1,-1),1, 2)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg + 1) # batch matrix-matrix product
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        return self.p_encoder, self.q_encoder

    def get_relevant_doc(self,
        query,
        k=1,
        args=None,
        p_encoder=None,
        q_encoder=None
        ):
    
        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder


        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query],
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(args.device)
            q_emb = q_encoder(**q_seqs_val)  # (num_query=1, emb_dim)
        self.p_embs = get_p_embs()
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", metavar="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        metavar="bert-base-multilingual-cased",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument(
        "--context_path", metavar="wikipedia_documents", type=str, help=""
    )
    parser.add_argument("--use_faiss", metavar=False, type=bool, help="")
    

    args = parser.parse_args()