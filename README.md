







# Read Me :D


## 👩먼저 해야할 일들

- ```zsh
  https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000077/data/data.tar.gz
  ```

- 위 데이터를 다운로드하고, git repo와 같은 경로에 압축을 풀어주세요!

- ```powershell
  # 이렇게 경로를 만들어주시면 됩니다 :D
  ┬📁data
  ├💻toolkits.py
  ├💻get_model_and_dataset.py
  ├💻train.py
  ├💻test.py
  └💻hardvoting.py
  ```

- ```zsh
  $ python -m pip install -r req.txt
  # 의존성 라이브러리들을 설치해주세요:D
  ```



## 📁./get_model_and_dataset.py

```zsh
$ python get_model_and_dataset.py 
```

> - `model.pt`, `tokenizer.pt`, `train_dataset.pt`, `test_dataset.pt` , `retrieved_context_list.json`파일들을 생성해서 `--output_path` 인자로 전달된 경로에 저장합니다.
> - `retrieved_context_list.json`은 test_dataset의 question들을 엘라스틱서치로 검색해서 얻은 결과들입니다.

- #### --name_or_path

  - 기본값 : `klue/roberta-large`

  - 1허깅페이스 모델명을 입력합니다. 기본값은 `klue/roberta-large`입니다.

- #### --train_dataset_path

  - 기본값 : `./data/train_dataset` 
  - 학습용 데이터셋이 저장된 path를 입력합니다.

- #### --test_dataset_path

  - 기본값 : `./data/test_dataset` 
  - 테스트용 데이터셋이 저장된 path를 입력합니다.

- #### --output_path

  - 기본값 : `./bin`
  - 피클 파일들을 저장할 위치를 입력합니다.





## 📁./train.py

```bash
$ python train.py
```

- #### --model_cp_path

  - 기본값 : `./bin/model.pt` 

- #### --tokenizer_cp_path
  - 기본값 : `./bin/tokenizer.pt` 

- #### --train_dataset_cp_path
  - 기본값 : `./bin/train_dataset.pt` 

- #### --valid_dataset_cp_path

  - 기본값 : `./bin/valid_dataset.pt` 

- #### --batch_size

  - 기본값 : `8`

- #### --lr

  - 기본값 : `0.00003`

- #### --warmup_steps

  - 기본값 : `200`

- #### --epoch

  - 기본값 : `10`

- #### --output_path 

  - 기본값 : `./train_results`

- #### no_token_type_ids

  - 기본값 : `True`
  - **klue-bert**처럼 `token_type_ids`를 입력으로 받는 모델의 경우,
  - False 옵션을 줘야합니다.





## 📁./test.py

```zsh
$ python test.py
```

- #### --model_cp_path

  - 기본값 : `./train_results/cp_list/*.pt`
  - 별도의 값을 설정하지 않으면 `./train_results/cp_list` 폴더에 있는 `.pt` 확장자 파일 중 하나를 model로 임포트합니다. 

- #### --tokenizer_cp_path

  - 기본값 : `./bin/tokenizer.pt` 

- #### --test_dataset_cp_path

  - 기본값 : `./data/test_dataset.pt` 

- #### --context_list_path

  - 기본값 : `./bin/retrieved_context_list.json` 

  - test_dataset과 연관된 문서들을 모아둔 파일입니다.
  - 엘라스틱 서치를 통해 검색한 결과입니다.

- #### --output_path

  - 기본값 : `./test_results`
  - 모델이 추론한 결과가 json 형태로 저장됩니다. 

- #### no_token_type_ids

  - 기본값 : `True`
  - **klue-bert**처럼 `token_type_ids`를 입력으로 받는 모델의 경우,
  - False 옵션을 줘야합니다.





## 📁./hardvoting.py

```zsh
$ python hardvoting.py
```

> `test.py`로 추론해서 얻은 결과들을 하드보팅을 통해 앙상블합니다.

- #### --path 

  - 기본값 : `./test_results`
  - 앞서 `test.py`로 추론한 결과(`json`)들이 담겨있는 폴더를 입력합니다.

- #### --output_path

  - 기본값 : `./emsemble_results`
  - 앙상블 결과가 저장될 폴더를 입력합니다.





