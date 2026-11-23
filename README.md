# AI 생성 음식점 리뷰 탐지

한국어 음식점 리뷰에서 LLM이 쓴 리뷰를 사람 리뷰와 구별할 수 있는지, 그 점수가 데이터의 쉬운 흔적 때문은 아닌지 확인하는 실험 저장소입니다.

## 브랜치
| 브랜치 | 내용 |
|---|---|
| [`exp1`](../../tree/exp1) | **1차 실험** — 정제 없이 데이터를 넣었을 때 얼마나 쉽게 구별되는지, 어떤 흔적(지름길)이 있는지 점검 |
| [`exp2`](../../tree/exp2) | **2차 실험** — 1차에서 찾은 흔적을 정제한 공용 데이터셋(`exp2_dataset.csv`)으로 재평가. 1차 코드 포함 |

## 공통 조건
- 사람 리뷰: KR3 (2022년 이전 음식점 리뷰) — [Kaggle](https://www.kaggle.com/datasets/ninetyninenewton/kr3-korean-restaurant-reviews-with-ratings)
- AI 리뷰: `generated.jsonl` (GPT-4.1 mini 1,200 + Claude Haiku 4.5 1,200), `generated_clean_v2.jsonl` (Gemma3·Gemma4·Qwen3, 처음 보는 생성기 시험용)
- 분할: 학습 8 : 평가 2, seed 42

## 원본 데이터 준비
원본 데이터는 저장소에 넣지 않습니다. 아래 파일을 `data_raw/` 폴더에 넣거나 `DATA_DIR` 환경변수로 폴더를 지정하세요.
```
data_raw/
├── kr3_raw.tsv               # Kaggle KR3 (원본)
├── kr3.tsv                   # Kaggle KR3 (전처리본, 1차 비교용)
├── generated.jsonl           # 노션
└── generated_clean_v2.jsonl  # 노션
```

---

# 1차 실험 — 데이터 점검과 지름길 진단

정제 없이 사람(KR3) 5,000건과 AI 리뷰를 그대로 넣고, 일부러 단순한 모델 3개로 얼마나 쉽게 구별되는지 봅니다.

| 모델 | 설명 |
|---|---|
| 길이만 | 글자 수 하나로 로지스틱 회귀 |
| 표면특징15 | ㅋ·이모지·줄바꿈·~·!·말줄임·^^·숫자·영문·어미(습니다/요/마침표) 개수, 길이, 띄어쓰기 비율, 문장 수. **단어 내용은 보지 않음** |
| TF-IDF | 문자 2~5-gram + 로지스틱 회귀 |

## 실행
```bash
pip install pandas scikit-learn
python exp1_baseline.py kr3_raw.tsv   # 사람 = KR3 원본
python exp1_baseline.py kr3.tsv       # 사람 = KR3 전처리본
python exp1_samples.py                # 그룹별 예시 출력
```
`kr3_raw.tsv`는 리뷰 안에 줄바꿈이 있으므로 `pd.read_csv(path, sep="\t")`(따옴표 처리 기본값)로 읽어야 642,413건이 나옵니다. `quoting=QUOTE_NONE`으로 읽으면 151만 줄로 깨집니다.

## 주요 결과 (TF-IDF, 0.5 기준, 사람 = 원본)
| 조건 | AUC | 탐지율 | 오탐률 |
|---|---|---|---|
| gen1 같은 분포 | 0.992 | 0.928 | 0.024 |
| gen1 학습 → gen2 평가 | 0.959 | 0.700 | 0.024 |
| gen1 프롬프트 A 학습 → C 평가 | 0.912 | 0.334 | 0.001 |

- 표면 특징만으로 AUC 0.85(원본), 0.89(전처리본). 높은 점수의 상당 부분은 시험이 쉬워서 나온다.
- 전처리본 KR3는 ㅋ·이모지·~·^^가 0%여서 'ㅋ가 있으면 AI'라는 잘못된 단서가 생긴다.
- 원본 KR3는 줄바꿈(사람 27.7% vs AI 0.5%), 203자 크롤링 잘림(9,650건), 외국어 리뷰(5,326건)가 단서가 된다.
- 전체 결과: `results/exp1_raw_*`, `results/exp1_pre_*`

## 파일
| 파일 | 역할 |
|---|---|
| `common.py` | 로드, 표면 특징, 평가 지표, 모델 3종 |
| `exp1_baseline.py` | 1차 실험 |
| `exp1_samples.py` | 예시 출력 |
| `results/exp1_*` | 결과 CSV와 로그 |
| `notebooks/exp1_kaggle_notebook.ipynb` | Kaggle용 노트북 (정제 없음, 참고용). `make_exp1_notebook.py`로 생성 |

---

# 2차 실험 — 정제 데이터로 재평가

1차에서 찾은 흔적을 **사람·AI 양쪽에 같은 규칙으로** 지운 공용 데이터셋을 만들고 다시 평가합니다.

## 정제 규칙 (`exp2_clean.py`)
| 단계 | 규칙 | 대상 | 건수 |
|---|---|---|---|
| 정규화 | NFC, 줄바꿈→공백, 연속 공백 1칸, zero-width·NBSP 제거, 굽은 따옴표 통일, 마크다운 `**`·목록 기호 제거, 감싼 따옴표 제거 | 공통 | 줄바꿈 27.7% → 0% |
| F1 | 크롤링 잘림: 원문 200~206자이면서 `...`로 끝남 | 사람 | 9,730 |
| F2·F3 | 한자·가나 포함, 한글 비율 50% 미만, 2자 미만 | 공통 | 사람 6,424 |
| D1·D2 | 완전 중복, 준중복(한글·영숫자만 남긴 키) | 공통 | 사람 3,963 / gen1 58 / gen2 76 |
| D3 | KR3 리뷰와 같은 AI 글 | AI | gen1 9 / gen2 15 |
| S1 | AI 최대 길이(462자) 초과 사람 리뷰 제외, gen1 길이 분포에 맞춰 사람 5,000건 추출 | 사람 | |
| 분할 | 8:2, seed 42, 라벨·길이 구간·생성기·프롬프트 층화 | 공통 | |

## 공용 데이터셋 `data/exp2_dataset.csv`
| 구성 | 학습 | 평가 |
|---|---|---|
| `set=main` 사람 | 4,000 | 1,000 |
| `set=main` gen1 (GPT/Claude) | 1,866 | 467 |
| `set=holdout_gen2` (Gemma3/Gemma4/Qwen3) | 2,732 | 683 |

컬럼: `text, label(0 사람/1 AI), generator, prompt_type, rating, len, len_bucket, split, set`

**팀원은 이 파일의 `split` 그대로 학습/평가해 주세요.** 그래야 결과를 같은 표에서 비교할 수 있습니다.

## 실행
```bash
python exp2_clean.py   # 정제 → data/exp2_dataset.csv (이미 포함되어 있어 다시 만들 필요는 없음)
python exp2_run.py     # 단계별 정제 효과, 시나리오 6개, 표기 제거 ablation, 길이별, 오류 예시
```

## 주요 결과 (TF-IDF, 0.5 기준)
| 조건 | AUC | 탐지율 | 오탐률 |
|---|---|---|---|
| 정제 전 (S0) gen1 같은 분포 | 0.990 | 0.921 | 0.017 |
| **정제 후** gen1 같은 분포 | 0.988 | 0.923 | 0.025 |
| gen1 학습 → gen2 평가 | 0.957 | 0.704 | 0.025 |
| gen1 A 학습 → C 평가 | 0.908 | 0.344 | 0.011 |
| 20자 이하 (gen1 같은 분포) | 0.934 | 0.714 | 0.048 |

- 흔적을 지워도 TF-IDF 점수는 거의 그대로다. 표면 특징 모델만 0.830 → 0.795로 떨어졌다. 높은 점수는 데이터 흔적이 아니라 **AI 문체 자체의 차이** 때문이다.
- 약점은 일반화다. 다른 프롬프트(A→C)와 다른 생성기에서 탐지율이 크게 떨어지고, 20자 이하 리뷰도 약하다.
- 전체 결과: `results/exp2_*`

## Kaggle에서 Transformer 돌리기
1. `data/exp2_dataset.csv`를 Kaggle Dataset으로 업로드
2. `notebooks/exp2_kaggle_notebook.ipynb`를 Import
3. Add Input으로 데이터셋 추가, Settings에서 GPU T4 x2, Internet On
4. `MODEL_NAME`만 바꾸고 Run All. 시나리오 3개(같은 분포, gen1→gen2, A→C) 결과가 `exp2_results.csv`로 저장됨

## 데이터 뷰어 (`viewer/`)
사람/AI 리뷰를 나란히 보며 흔적을 색으로 표시하고, 정제 전후 비교와 맞혀보기를 할 수 있는 페이지입니다.
```bash
cd viewer && python -m http.server 8000   # 브라우저에서 http://localhost:8000
```
`index.html`을 파일로 바로 열면 `reviews.json`을 불러오지 못하니 위처럼 서버로 열어 주세요. 표본은 `python viewer/make_data.py`로 다시 만들 수 있습니다.

## 파일
| 파일 | 역할 |
|---|---|
| `exp2_clean.py` | 정제 규칙, 공용 데이터셋 생성 |
| `exp2_run.py` | 2차 평가 |
| `data/exp2_dataset.csv` | 공용 데이터셋 |
| `results/exp2_*` | 결과 CSV와 로그 |
| `notebooks/exp2_kaggle_notebook.ipynb` | Kaggle용 (TF-IDF + Transformer 미세조정). `make_exp2_notebook.py`로 생성 |
| `viewer/` | 데이터 뷰어 |
