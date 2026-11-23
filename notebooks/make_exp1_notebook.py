import nbformat as nbf
import sys

out = sys.argv[1]
nb = nbf.v4.new_notebook()
C = []

def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# AI 생성 음식점 리뷰 탐지 — Kaggle 노트북

**공통 조건**: 사람 리뷰 KR3 5,000건 / AI 리뷰 GPT-4.1 mini 1,200 + Claude Haiku 4.5 1,200 / 학습 8 : 평가 2, seed 42

순서: ① 설정 → ② 입력 파일 확인 → ③ 데이터 로드 → ④ 분할 → ⑤ 평가 함수 → ⑥ 베이스라인(TF-IDF) → ⑦ 본 모델(미세조정) → ⑧ 결과표 → ⑨ 오류 분석

> 시작 전: 오른쪽 패널 **Settings → Accelerator: GPU T4 x2**, **Internet: On** (Hugging Face 모델 다운로드용)
""")

md("## ① 설정 — 여기만 바꾸면 됩니다")
code(r'''
SEED = 42
N_HUMAN = 5000
TEST_SIZE = 0.2

INPUT_DIR = "/kaggle/input"
WORK_DIR = "/kaggle/working"

# AI 리뷰 파일 이름 (팀에서 쓰기로 한 것). 정제본이면 generated_clean_v2.jsonl
AI_FILE_NAME = "generated_clean_v2.jsonl"

# 팀이 공유하는 분할 파일이 있으면 경로를 넣으세요. 없으면 None → 직접 만들고 저장합니다.
SHARED_SPLIT_FILE = None   # 예: "/kaggle/input/team-split/split_seed42.csv"

# 미세조정할 모델 (다른 팀원과 겹치지 않게 고르세요)
#   klue/bert-base, monologg/koelectra-base-v3-discriminator, beomi/kcbert-base,
#   microsoft/mdeberta-v3-base, snunlp/KR-ELECTRA-discriminator ...
MODEL_NAME = "klue/bert-base"
MAX_LEN = 256
EPOCHS = 3
LR = 2e-5
BATCH = 32
THRESHOLD = 0.5
''')

md("## ② 입력 파일 확인")
code(r'''
import os, glob
files = sorted(p for p in glob.glob(f"{INPUT_DIR}/**/*", recursive=True) if os.path.isfile(p))
for p in files:
    print(f"{os.path.getsize(p)/1e6:8.1f} MB  {p}")
''')

md("## ③ 데이터 로드\n컬럼 이름은 자동으로 찾습니다. 출력된 컬럼/샘플을 꼭 눈으로 확인하세요.")
code(r'''
import re, random, unicodedata
import numpy as np, pandas as pd

random.seed(SEED); np.random.seed(SEED)

TEXT_CANDIDATES = ["Review", "review", "text", "content", "generated", "generated_text", "output", "body"]

def pick_text_col(df, where):
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            return c
    obj = [c for c in df.columns if df[c].dtype == object]
    if not obj:
        raise ValueError(f"{where}: 텍스트 컬럼을 못 찾았습니다. 컬럼={list(df.columns)}")
    # 평균 길이가 가장 긴 문자열 컬럼
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean())

def read_table(path):
    if path.endswith(".parquet"): return pd.read_parquet(path)
    if path.endswith(".jsonl") or path.endswith(".json"): return pd.read_json(path, lines=path.endswith(".jsonl"))
    if path.endswith(".tsv"): return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)

def normalize(s):
    s = unicodedata.normalize("NFC", str(s))
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s.strip()

# ---- 사람 리뷰 (KR3) ----
kr3_files = [p for p in files if "kr3" in os.path.basename(p).lower()
             and p.endswith((".parquet", ".tsv", ".csv"))]
kr3_files.sort(key=lambda p: (not p.endswith(".parquet"), "raw" in p.lower()))  # parquet 우선, 전처리본 우선
assert kr3_files, "KR3 파일을 못 찾았습니다. Add Input으로 KR3 데이터셋을 추가했는지 확인하세요."
kr3_path = kr3_files[0]
print("KR3:", kr3_path)
kr3 = read_table(kr3_path)
print(kr3.columns.tolist(), len(kr3))
h_col = pick_text_col(kr3, "KR3")

human = pd.DataFrame({"text": kr3[h_col].map(normalize)})
human = human[human.text.str.len() > 0].drop_duplicates("text")
human = human.sample(n=min(N_HUMAN, len(human)), random_state=SEED).reset_index(drop=True)
human["label"] = 0
human["source"] = "human"

# ---- AI 리뷰 ----
ai_files = [p for p in files if os.path.basename(p) == AI_FILE_NAME]
assert ai_files, f"{AI_FILE_NAME} 을 못 찾았습니다. 직접 업로드한 데이터셋을 Add Input 했는지 확인하세요."
ai_raw = read_table(ai_files[0])
print("AI:", ai_files[0], ai_raw.columns.tolist(), len(ai_raw))
a_col = pick_text_col(ai_raw, "AI")
gen_col = next((c for c in ["generator", "model", "llm", "source", "provider"] if c in ai_raw.columns), None)

ai = pd.DataFrame({"text": ai_raw[a_col].map(normalize)})
ai["source"] = ai_raw[gen_col].astype(str) if gen_col else "ai"
ai = ai[ai.text.str.len() > 0].drop_duplicates("text").reset_index(drop=True)
ai["label"] = 1

df = pd.concat([human, ai], ignore_index=True).drop_duplicates("text", keep=False).reset_index(drop=True)

def length_bucket(n):
    if n <= 20: return "≤20"
    if n <= 50: return "21–50"
    if n <= 150: return "51–150"
    return ">150"
df["len"] = df.text.str.len()
df["len_bucket"] = df["len"].map(length_bucket)

print(f"\n텍스트 컬럼: 사람={h_col!r}, AI={a_col!r}, 생성기 컬럼={gen_col!r}")
print(df.groupby(["label", "source"]).size())
display(df.groupby("label").sample(3, random_state=SEED)[["label", "source", "text"]])
''')

md("## ④ 학습 8 : 평가 2 분할 (seed 42)\n팀원끼리 **같은 평가셋**이어야 표의 숫자를 비교할 수 있습니다. 한 명이 만든 `split_seed42.csv`를 Kaggle Dataset으로 공유해서 `SHARED_SPLIT_FILE`에 넣는 것을 권장합니다.")
code(r'''
from sklearn.model_selection import train_test_split

if SHARED_SPLIT_FILE:
    split = pd.read_csv(SHARED_SPLIT_FILE)
    split["text"] = split["text"].map(normalize)
    df = df.drop(columns=[c for c in ["split"] if c in df.columns]).merge(split[["text", "split"]], on="text", how="inner")
    print("공유 분할 사용:", df.split.value_counts().to_dict())
else:
    tr_idx, te_idx = train_test_split(df.index, test_size=TEST_SIZE, random_state=SEED,
                                      stratify=df.label.astype(str) + "_" + df.len_bucket)
    df["split"] = "train"
    df.loc[te_idx, "split"] = "test"
    df[["text", "label", "source", "len_bucket", "split"]].to_csv(f"{WORK_DIR}/split_seed42.csv", index=False)
    print("분할 저장:", f"{WORK_DIR}/split_seed42.csv")

train_df = df[df.split == "train"].reset_index(drop=True)
test_df = df[df.split == "test"].reset_index(drop=True)
print(pd.crosstab(df.split, df.label, margins=True))
''')

md("## ⑤ 평가 함수\n팀 표 형식: 정확도 / F1 / 탐지율(TPR, AI를 AI로 잡은 비율) / 오탐률(FPR, 사람을 AI로 잘못 잡은 비율). AUC도 같이 봅니다.")
code(r'''
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def metrics(y, prob, thr=THRESHOLD):
    y = np.asarray(y); prob = np.asarray(prob)
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": len(y),
        "정확도": accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "탐지율": tp / (tp + fn) if tp + fn else np.nan,
        "오탐률": fp / (fp + tn) if fp + tn else np.nan,
        "AUC": roc_auc_score(y, prob) if len(set(y)) == 2 else np.nan,
    }

def report(name, test_df, prob, thr=THRESHOLD):
    rows = [{"모델": name, "길이 조건": "전체", **metrics(test_df.label, prob, thr)}]
    for b in ["≤20", "21–50", "51–150", ">150"]:
        m = test_df.len_bucket.values == b
        if m.any():
            rows.append({"모델": name, "길이 조건": b, **metrics(test_df.label[m], prob[m], thr)})
    out = pd.DataFrame(rows)
    return out

def pretty(r):
    r = r.copy()
    for c in ["정확도", "탐지율", "오탐률"]:
        r[c] = (r[c] * 100).map(lambda v: f"{v:.1f}%")
    for c in ["F1", "AUC"]:
        r[c] = r[c].map(lambda v: f"{v:.3f}")
    return r

results = []
''')

md("## ⑥ 베이스라인: TF-IDF(문자 n-gram) + 로지스틱 회귀\nCPU로 1분 안에 끝납니다. 본 모델이 이것보다 나은지 비교하는 기준선입니다.")
code(r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(train_df.text); Xte = vec.transform(test_df.text)
lr = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=SEED)
lr.fit(Xtr, train_df.label)
prob_tfidf = lr.predict_proba(Xte)[:, 1]

r = report("TF-IDF + LR", test_df, prob_tfidf); results.append(r)
display(pretty(r))
''')

md("## ⑦ 본 모델: Transformer 미세조정\nGPU T4 기준 base 모델 3 epoch ≈ 5~10분. 평가셋은 학습/모델 선택에 쓰지 않습니다(고정 epoch).")
code(r'''
import torch, os
os.environ["WANDB_DISABLED"] = "true"
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding, set_seed)
set_seed(SEED)
print("GPU:", torch.cuda.is_available(), torch.cuda.device_count())

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def to_ds(d):
    ds = Dataset.from_pandas(d[["text", "label"]].rename(columns={"label": "labels"}))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])
ds_tr, ds_te = to_ds(train_df), to_ds(test_df)

args = TrainingArguments(
    output_dir=f"{WORK_DIR}/ckpt",
    num_train_epochs=EPOCHS, learning_rate=LR, weight_decay=0.01, warmup_ratio=0.1,
    per_device_train_batch_size=BATCH // max(1, torch.cuda.device_count()),
    per_device_eval_batch_size=64,
    fp16=torch.cuda.is_available(),
    save_strategy="no", logging_steps=50, report_to="none", seed=SEED,
)
trainer = Trainer(model=model, args=args, train_dataset=ds_tr,
                  data_collator=DataCollatorWithPadding(tok))
trainer.train()

logits = trainer.predict(ds_te).predictions
prob_ft = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()

r = report(MODEL_NAME, test_df, prob_ft); results.append(r)
display(pretty(r))
''')

md("## ⑧ 결과표 저장 (팀 표에 옮겨 적을 값)")
code(r'''
res = pd.concat(results, ignore_index=True)
res.to_csv(f"{WORK_DIR}/results.csv", index=False)
pd.DataFrame({"text": test_df.text, "label": test_df.label, "source": test_df.source,
              "len_bucket": test_df.len_bucket, "prob_tfidf": prob_tfidf, "prob_ft": prob_ft}
             ).to_csv(f"{WORK_DIR}/test_predictions.csv", index=False)
display(pretty(res))
''')

md("## ⑨ 오류 분석 — 놓친 AI 리뷰 / 사람인데 AI로 잡힌 리뷰")
code(r'''
pred_df = test_df.assign(prob=prob_ft)
print("■ 놓친 AI 리뷰 (점수 낮은 순)")
display(pred_df[(pred_df.label == 1) & (pred_df.prob < THRESHOLD)].sort_values("prob").head(15)[["prob", "source", "text"]])
print("■ 오탐된 사람 리뷰 (점수 높은 순)")
display(pred_df[(pred_df.label == 0) & (pred_df.prob >= THRESHOLD)].sort_values("prob", ascending=False).head(15)[["prob", "text"]])
print("■ 생성기별 탐지율")
display(pred_df[pred_df.label == 1].groupby("source").prob.apply(lambda p: (p >= THRESHOLD).mean()))
''')

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, out)
print("wrote", out)
