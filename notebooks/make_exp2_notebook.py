"""2차 실험 Kaggle 노트북 생성: python make_exp2_notebook.py exp2_kaggle_notebook.ipynb"""
import nbformat as nbf, sys
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# AI 생성 음식점 리뷰 탐지 — 2차 실험 (정제 데이터)

`exp2_dataset.csv` 하나로 모든 팀원이 **같은 학습/평가 분할**을 씁니다.
- `set=main`: 사람(KR3 원본, 정제·길이 맞춤) 5,000 + GPT/Claude 2,333
- `set=holdout_gen2`: Gemma3/Gemma4/Qwen3 3,415 (처음 보는 생성기 시험용)
- `split=train/test`: 8:2, seed 42, 라벨·길이 구간·생성기·프롬프트 층화

정제 규칙은 `exp2_clean.py` 참고(줄바꿈→공백, 잘린 리뷰·외국어 제거, 중복·준중복·교차 중복 제거, 길이 분포 맞춤).

> 시작 전: Add Input으로 `exp2_dataset.csv`가 든 데이터셋 추가, Settings → **GPU T4 x2**, **Internet On**
""")
md("## ① 설정")
code(r'''
SEED = 42
DATA_PATH = None          # None이면 /kaggle/input 아래에서 exp2_dataset.csv를 자동으로 찾음
WORK_DIR = "/kaggle/working"
MODEL_NAME = "klue/bert-base"   # 팀원과 겹치지 않게 선택
MAX_LEN, EPOCHS, LR, BATCH, THRESHOLD = 256, 3, 2e-5, 32, 0.5
''')
md("## ② 데이터 로드")
code(r'''
import glob, numpy as np, pandas as pd
if DATA_PATH is None:
    DATA_PATH = glob.glob("/kaggle/input/**/exp2_dataset.csv", recursive=True)[0]
df = pd.read_csv(DATA_PATH)
H  = df[(df.set == "main") & (df.label == 0)]
G1 = df[(df.set == "main") & (df.label == 1)]
G2 = df[df.set == "holdout_gen2"]
tr_ = lambda d: d[d.split == "train"]; te_ = lambda d: d[d.split == "test"]
cat = lambda *a: pd.concat(a, ignore_index=True)
print(pd.crosstab([df.set, df.generator], df.split))
''')
md("## ③ 평가 함수\n정확도는 사람:AI = 2:1이라 부풀려지니 **AUC·탐지율·오탐률**을 같이 보고하세요. TPR@FPR5는 평가셋으로 기준선을 맞춘 이상적 값(oracle)입니다.")
code(r'''
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
def metrics(y, p, thr=THRESHOLD):
    y, p = np.asarray(y), np.asarray(p); pr = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    return {"n_human": int((y==0).sum()), "n_ai": int((y==1).sum()),
            "정확도": accuracy_score(y, pr), "F1": f1_score(y, pr, zero_division=0),
            "탐지율": pr[y==1].mean(), "오탐률": pr[y==0].mean(), "AUC": roc_auc_score(y, p),
            "TPR@FPR5(oracle)": float(np.interp(0.05, fpr, tpr))}
def by_length(te, p):
    rows = [{"길이": "전체", **metrics(te.label, p)}]
    for b in ["≤20", "21–50", "51–150", ">150"]:
        m = (te.len_bucket == b).values
        rows.append({"길이": b, **metrics(te.label.values[m], p[m])})
    return pd.DataFrame(rows).round(3)
''')
md("## ④ 시나리오 정의 (2차 실험과 동일)")
code(r'''
SCENARIOS = {
    "gen1 같은 분포":        (cat(tr_(H), tr_(G1)), cat(te_(H), te_(G1))),
    "gen1 학습 → gen2 평가": (cat(tr_(H), tr_(G1)), cat(te_(H), G2)),
    "gen1 A 학습 → C 평가":  (cat(tr_(H), tr_(G1)[tr_(G1).prompt_type == "A"]), cat(te_(H), G1[G1.prompt_type == "C"])),
}
''')
md("## ⑤ 베이스라인: TF-IDF + LR (CPU, 1분)")
code(r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
def tfidf(tr, te):
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    m = LogisticRegression(C=4, max_iter=3000, class_weight="balanced")
    m.fit(v.fit_transform(tr.text), tr.label); return m.predict_proba(v.transform(te.text))[:, 1]
results = []
for name, (tr, te) in SCENARIOS.items():
    results.append({"모델": "TF-IDF + LR", "조건": name, **metrics(te.label, tfidf(tr, te))})
display(pd.DataFrame(results).round(3))
''')
md("## ⑥ 본 모델: Transformer 미세조정 (시나리오마다 새로 학습)")
code(r'''
import os, torch
os.environ["WANDB_DISABLED"] = "true"
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding, set_seed)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def to_ds(d):
    ds = Dataset.from_pandas(d[["text", "label"]].rename(columns={"label": "labels"}).reset_index(drop=True))
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN), batched=True, remove_columns=["text"])
def finetune(tr, te):
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    args = TrainingArguments(output_dir=f"{WORK_DIR}/ckpt", num_train_epochs=EPOCHS, learning_rate=LR,
        weight_decay=0.01, warmup_ratio=0.1, per_device_train_batch_size=BATCH // max(1, torch.cuda.device_count()),
        per_device_eval_batch_size=64, fp16=torch.cuda.is_available(), save_strategy="no",
        logging_steps=50, report_to="none", seed=SEED)
    t = Trainer(model=model, args=args, train_dataset=to_ds(tr), data_collator=DataCollatorWithPadding(tok))
    t.train()
    logits = t.predict(to_ds(te)).predictions
    return torch.softmax(torch.tensor(logits), -1)[:, 1].numpy()

preds = {}
for name, (tr, te) in SCENARIOS.items():
    print("▶", name); preds[name] = finetune(tr, te)
    results.append({"모델": MODEL_NAME, "조건": name, **metrics(te.label, preds[name])})
res = pd.DataFrame(results).round(3); display(res)
''')
md("## ⑦ 저장 · 길이 구간별 · 오류 분석")
code(r'''
res.to_csv(f"{WORK_DIR}/exp2_results.csv", index=False)
tr, te = SCENARIOS["gen1 같은 분포"]; p = preds["gen1 같은 분포"]
display(by_length(te.reset_index(drop=True), p))
te = te.reset_index(drop=True).assign(p=p)
te.to_csv(f"{WORK_DIR}/exp2_test_predictions.csv", index=False)
print("■ 놓친 AI 글"); display(te[(te.label==1)&(te.p<THRESHOLD)].sort_values("p").head(15)[["p","generator","prompt_type","text"]])
print("■ 오탐된 사람 글"); display(te[(te.label==0)&(te.p>=THRESHOLD)].sort_values("p",ascending=False).head(15)[["p","text"]])
''')
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
nbf.write(nb, sys.argv[1])
