import torch
import wandb
from transformers.hf_argparser import HfArgumentParser
from dataclasses import dataclass
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from toolkits import TrainQAModel


@dataclass
class TrainArguments:
    model_cp_path: str = "./bin/model.pt"
    tokenizer_cp_path: str = "./bin/tokenizer.pt"
    train_dataset_cp_path: str = "./bin/train_dataset.pt"
    valid_dataset_cp_path: str = "./bin/valid_dataset.pt"
    batch_size: int = 8
    lr: float = 0.00003
    warmup_steps: int = 200
    epoch: int = 10
    output_path = "./train_results/"
    no_token_type_ids = True


parser = HfArgumentParser([TrainArguments])
(model_args,) = parser.parse_args_into_dataclasses()

model = torch.load(model_args.model_cp_path)
tokenizer = torch.load(model_args.tokenizer_cp_path)
train_dataset = torch.load(model_args.train_dataset_cp_path)
valid_dataset = torch.load(model_args.valid_dataset_cp_path)

trainer = TrainQAModel(
    model=model,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    output_path=model_args.output_path,
    no_token_type_ids=model_args.no_token_type_ids,
    lr=model_args.lr,
    warmup_steps=model_args.warmup_steps,
    batch_size=model_args.batch_size,
    epoch=model_args.epoch,
)
for i in range(model_args.epoch):
    print(f"{i+1}λ²μ§Έ πνμ΅μ μμν©λλ€.")
    trainer.train()
    print(f"{i+1}λ²μ§Έπκ²μ¦μ μμν©λλ€.")
    trainer.valid()
    print("πμ²΄ν¬ν¬μΈνΈλ₯Ό μ μ₯ν©λλ€.")
    trainer.save_cp()
    print("πλ² μ€νΈ λͺ¨λΈλ§ λ¨κΈ°κ³  μ λΆ μ­μ ν©λλ€.")
    trainer.extract_best_model()
    if trainer.valid_acc < 0.01:
        print("π€¦ββοΈμ΄κ±΄ λ§ν λͺ¨λΈμλλ€. νμ΅μ μ€λ¨ν©λλ€.")
        break
