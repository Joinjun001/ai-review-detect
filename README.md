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
