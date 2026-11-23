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
