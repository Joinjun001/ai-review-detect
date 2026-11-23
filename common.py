"""1차·2차 실험 공통 함수: 데이터 로드, 표면 특징, 평가."""
import os, re, unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

SEED = 42
# 원본 데이터 폴더: 저장소의 data_raw/ 에 넣거나, DATA_DIR 환경변수로 지정
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_raw"))
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------- 로드 ----------
def load_kr3(name="kr3_raw.tsv"):
    """kr3_raw.tsv는 리뷰 안에 줄바꿈이 있어 따옴표 처리를 켜고(기본값) 읽어야 642,413건이 나온다."""
    d = pd.read_csv(os.path.join(DATA_DIR, name), sep="\t", dtype=str)
    d = d[d.Review.notna()].copy()
    d["text"] = d.Review
    return d


def load_ai(name):
    d = pd.read_json(os.path.join(DATA_DIR, name), lines=True)
    d = d[d.status == "ok"].copy()
    return d


def nfc(s):
    return unicodedata.normalize("NFC", str(s)).strip()


# ---------- 표면 특징 ----------
SURFACE = {
    "ㅋ/ㅎ/ㅠ 낱자": r"[ㄱ-ㅎㅏ-ㅣ]",
    "이모지": r"[\U0001F300-\U0001FAFF☀-➿]",
    "줄바꿈": r"\n",
    "물결 ~": r"~",
    "느낌표 !": r"!",
    "말줄임 ..": r"\.\.|…",
    "^^": r"\^\^",
    "숫자": r"\d",
    "영문": r"[A-Za-z]",
    "끝이 -습니다/니다": r"니다[.!~ ]*$",
    "끝이 -요": r"요[.!~^ ]*$",
    "끝이 마침표 .": r"\.$",
}


def surface_profile(t: pd.Series):
    r = {k: t.str.contains(v, regex=True).mean() * 100 for k, v in SURFACE.items()}
    r["평균 길이"] = t.str.len().mean()
    r["중앙 길이"] = t.str.len().median()
    return r


def surface_X(t: pd.Series):
    X = pd.DataFrame({k: t.str.count(v) for k, v in SURFACE.items()})
    L = t.str.len()
    X["log_len"] = np.log1p(L)
    X["space_ratio"] = t.str.count(" ") / L.clip(lower=1)
    X["n_sent"] = t.str.count(r"[.!?]")
    return X.values


# ---------- 평가 ----------
def metrics(y, p, thr=0.5):
    y = np.asarray(y); p = np.asarray(p); pr = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    return {
        "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
        "acc": accuracy_score(y, pr), "F1": f1_score(y, pr, zero_division=0),
        "TPR": pr[y == 1].mean(), "FPR": pr[y == 0].mean(),
        "AUC": roc_auc_score(y, p),
        # 평가셋으로 기준선을 맞춘 이상적 값(oracle) — 실제 운영에서는 얻을 수 없음
        "TPR@FPR5(oracle)": float(np.interp(0.05, fpr, tpr)),
    }


def fit_predict(model, tr, te):
    if model == "길이만":
        m = LogisticRegression(max_iter=1000)
        f = lambda t: np.log1p(t.str.len().values)[:, None]
        m.fit(f(tr.text), tr.y); return m.predict_proba(f(te.text))[:, 1]
    if model == "표면특징15":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        m.fit(surface_X(tr.text), tr.y); return m.predict_proba(surface_X(te.text))[:, 1]
    if model == "TF-IDF":
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
        m = LogisticRegression(C=4, max_iter=3000, class_weight="balanced")
        m.fit(v.fit_transform(tr.text), tr.y); return m.predict_proba(v.transform(te.text))[:, 1]
    raise ValueError(model)


MODELS = ["길이만", "표면특징15", "TF-IDF"]


def run_scenario(human, ai_train, ai_test=None, tag="", models=MODELS, strat_h=None, strat_a=None):
    """사람은 항상 8:2 분할. ai_test가 None이면 AI도 8:2(같은 분포), 아니면 ai_train의 학습 80%로 학습 후 ai_test 전체로 평가."""
    h_tr, h_te = train_test_split(human, test_size=0.2, random_state=SEED, stratify=strat_h)
    a_tr, a_te = train_test_split(ai_train, test_size=0.2, random_state=SEED, stratify=strat_a)
    if ai_test is not None:
        a_te = ai_test
    tr = pd.concat([h_tr.assign(y=0), a_tr.assign(y=1)], ignore_index=True)
    te = pd.concat([h_te.assign(y=0), a_te.assign(y=1)], ignore_index=True)
    rows = []
    for m in models:
        p = fit_predict(m, tr, te)
        rows.append({"조건": tag, "모델": m, **metrics(te.y, p)})
    return pd.DataFrame(rows), te, p


def fmt(df):
    d = df.copy()
    for c in ["acc", "F1", "TPR", "FPR", "AUC", "TPR@FPR5(oracle)"]:
        if c in d: d[c] = d[c].map(lambda v: f"{v:.3f}")
    return d
