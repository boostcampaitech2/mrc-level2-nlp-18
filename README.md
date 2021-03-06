







# Read Me :D


## ๐ฉ๋จผ์  ํด์ผํ  ์ผ๋ค
- ```zsh
  https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/data.tar.gz
  ```

- ์ ๋ฐ์ดํฐ๋ฅผ ๋ค์ด๋ก๋ํ๊ณ , git repo์ ๊ฐ์ ๊ฒฝ๋ก์ ์์ถ์ ํ์ด์ฃผ์ธ์!
- ํ์ด์ฌ์ 3.7.1 ๋ฒ์ ์ผ๋ก ์ฌ์ฉํด์ฃผ์ธ์ :D

- ```powershell
  # ์ด๋ ๊ฒ ๊ฒฝ๋ก๋ฅผ ๋ง๋ค์ด์ฃผ์๋ฉด ๋ฉ๋๋ค :D
  โฌ๐data
  โ๐ปtoolkits.py
  โ๐ปget_model_and_dataset.py
  โ๐ปtrain.py
  โ๐ปtest.py
  โ๐ปhardvoting.py
  ```

- ```zsh
  $ python -m pip install -r req.txt
  # ์์กด์ฑ ๋ผ์ด๋ธ๋ฌ๋ฆฌ๋ค์ ์ค์นํด์ฃผ์ธ์:D
  ```



## ๐./get_model_and_dataset.py

```zsh
$ python get_model_and_dataset.py 
```

> - `model.pt`, `tokenizer.pt`, `train_dataset.pt`, `test_dataset.pt` , `retrieved_context_list.json`ํ์ผ๋ค์ ์์ฑํด์ `--output_path` ์ธ์๋ก ์ ๋ฌ๋ ๊ฒฝ๋ก์ ์ ์ฅํฉ๋๋ค.
> - `retrieved_context_list.json`์ test_dataset์ question๋ค์ ์๋ผ์คํฑ์์น๋ก ๊ฒ์ํด์ ์ป์ ๊ฒฐ๊ณผ๋ค์๋๋ค.

- #### --name_or_path

  - ๊ธฐ๋ณธ๊ฐ : `klue/roberta-large`

  - 1ํ๊นํ์ด์ค ๋ชจ๋ธ๋ช์ ์๋ ฅํฉ๋๋ค. ๊ธฐ๋ณธ๊ฐ์ `klue/roberta-large`์๋๋ค.

- #### --train_dataset_path

  - ๊ธฐ๋ณธ๊ฐ : `./data/train_dataset` 
  - ํ์ต์ฉ ๋ฐ์ดํฐ์์ด ์ ์ฅ๋ path๋ฅผ ์๋ ฅํฉ๋๋ค.

- #### --test_dataset_path

  - ๊ธฐ๋ณธ๊ฐ : `./data/test_dataset` 
  - ํ์คํธ์ฉ ๋ฐ์ดํฐ์์ด ์ ์ฅ๋ path๋ฅผ ์๋ ฅํฉ๋๋ค.

- #### --output_path

  - ๊ธฐ๋ณธ๊ฐ : `./bin`
  - ํผํด ํ์ผ๋ค์ ์ ์ฅํ  ์์น๋ฅผ ์๋ ฅํฉ๋๋ค.





## ๐./train.py

```bash
$ python train.py
```

- #### --model_cp_path

  - ๊ธฐ๋ณธ๊ฐ : `./bin/model.pt` 

- #### --tokenizer_cp_path
  - ๊ธฐ๋ณธ๊ฐ : `./bin/tokenizer.pt` 

- #### --train_dataset_cp_path
  - ๊ธฐ๋ณธ๊ฐ : `./bin/train_dataset.pt` 

- #### --valid_dataset_cp_path

  - ๊ธฐ๋ณธ๊ฐ : `./bin/valid_dataset.pt` 

- #### --batch_size

  - ๊ธฐ๋ณธ๊ฐ : `8`

- #### --lr

  - ๊ธฐ๋ณธ๊ฐ : `0.00003`

- #### --warmup_steps

  - ๊ธฐ๋ณธ๊ฐ : `200`

- #### --epoch

  - ๊ธฐ๋ณธ๊ฐ : `10`

- #### --output_path 

  - ๊ธฐ๋ณธ๊ฐ : `./train_results`

- #### --no_token_type_ids

  - ๊ธฐ๋ณธ๊ฐ : `True`
  - **klue-bert**์ฒ๋ผ `token_type_ids`๋ฅผ ์๋ ฅ์ผ๋ก ๋ฐ๋ ๋ชจ๋ธ์ ๊ฒฝ์ฐ,
  - False ์ต์์ ์ค์ผํฉ๋๋ค.





## ๐./test.py

```zsh
$ python test.py
```

- #### --model_cp_path

  - ๊ธฐ๋ณธ๊ฐ : `./train_results/cp_list/*.pt`
  - ๋ณ๋์ ๊ฐ์ ์ค์ ํ์ง ์์ผ๋ฉด `./train_results/cp_list` ํด๋์ ์๋ `.pt` ํ์ฅ์ ํ์ผ ์ค ํ๋๋ฅผ model๋ก ์ํฌํธํฉ๋๋ค. 

- #### --tokenizer_cp_path

  - ๊ธฐ๋ณธ๊ฐ : `./bin/tokenizer.pt` 

- #### --test_dataset_cp_path

  - ๊ธฐ๋ณธ๊ฐ : `./data/test_dataset.pt` 

- #### --context_list_path

  - ๊ธฐ๋ณธ๊ฐ : `./bin/retrieved_context_list.json` 

  - test_dataset๊ณผ ์ฐ๊ด๋ ๋ฌธ์๋ค์ ๋ชจ์๋ ํ์ผ์๋๋ค.
  - ์๋ผ์คํฑ ์์น๋ฅผ ํตํด ๊ฒ์ํ ๊ฒฐ๊ณผ์๋๋ค.

- #### --output_path

  - ๊ธฐ๋ณธ๊ฐ : `./test_results`
  - ๋ชจ๋ธ์ด ์ถ๋ก ํ ๊ฒฐ๊ณผ๊ฐ json ํํ๋ก ์ ์ฅ๋ฉ๋๋ค. 

- #### --no_token_type_ids

  - ๊ธฐ๋ณธ๊ฐ : `True`
  - **klue-bert**์ฒ๋ผ `token_type_ids`๋ฅผ ์๋ ฅ์ผ๋ก ๋ฐ๋ ๋ชจ๋ธ์ ๊ฒฝ์ฐ,
  - False ์ต์์ ์ค์ผํฉ๋๋ค.





## ๐./hardvoting.py

```zsh
$ python hardvoting.py
```

> `test.py`๋ก ์ถ๋ก ํด์ ์ป์ ๊ฒฐ๊ณผ๋ค์ ํ๋๋ณดํ์ ํตํด ์์๋ธํฉ๋๋ค.

- #### --path 

  - ๊ธฐ๋ณธ๊ฐ : `./test_results`
  - ์์ `test.py`๋ก ์ถ๋ก ํ ๊ฒฐ๊ณผ(`json`)๋ค์ด ๋ด๊ฒจ์๋ ํด๋๋ฅผ ์๋ ฅํฉ๋๋ค.

- #### --output_path

  - ๊ธฐ๋ณธ๊ฐ : `./emsemble_results`
  - ์์๋ธ ๊ฒฐ๊ณผ๊ฐ ์ ์ฅ๋  ํด๋๋ฅผ ์๋ ฅํฉ๋๋ค.




 
