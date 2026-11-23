"""2차 실험 데이터 정제.

1차 실험에서 찾은 '내용과 무관하게 사람/AI를 가르는 흔적'을 두 클래스에 같은 규칙으로 없앤다.

  [정규화 — 사람·AI 공통]
   N1 NFC, 줄바꿈 → 공백, 연속 공백 1칸, 보이지 않는 문자(zero-width, NBSP) 제거
   N2 굽은 따옴표 → 곧은 따옴표, 마크다운 굵게(**), 줄머리 목록 기호(- * •) 제거, 글 전체를 감싼 따옴표 제거
  [필터]
   F1 (사람) 크롤링 미리보기로 잘린 리뷰: 원문 200~206자 이면서 '...'로 끝남
   F2 (공통) 한자·가나 포함, 또는 한글 비율 50% 미만(영어 위주) 리뷰
   F3 (공통) 2자 미만
  [중복]
   D1 (공통) 정규화 후 완전 중복
   D2 (공통) 준중복: 한글·영숫자만 남긴 키가 같은 글
   D3 (교차) AI 글과 같은 키를 가진 KR3 리뷰가 있으면 AI 글 제거
  [샘플링]
   S1 사람 5,000건을 AI(gen1) 길이 분포(≤20/21–50/51–150/>150)에 맞춰 추출. AI 최대 길이보다 긴 사람 리뷰는 제외
  [분할]
   8:2, seed 42, 라벨·길이 구간(·생성기·프롬프트) 층화

출력: data/exp2_dataset.csv  (set=main: 사람+gen1 / set=holdout_gen2: gen2, 각각 split=train/test)
      results/exp2_clean_log.txt
"""
import os, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from common import *

N_HUMAN = 5000
BINS = [0, 20, 50, 150, 10**6]
BIN_LABELS = ["≤20", "21–50", "51–150", ">150"]

ZW = re.compile(r"[​‌‍﻿ ]")
CJK = re.compile(r"[一-鿿぀-ヿ]")
HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")
LATIN = re.compile(r"[A-Za-z]")


def normalize(s, level=2):
    s = nfc(s)
    if level >= 1:
        s = ZW.sub(" ", s)
        s = re.sub(r"\r\n?|\n", " ", s)
    if level >= 2:
        s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        s = s.replace("**", "")
        s = re.sub(r"(^|\s)[-*•]\s+(?=\S)", r"\1", s)  # 목록 기호
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
            s = s[1:-1]
    if level >= 1:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def key(s):
    return re.sub(r"[^가-힣a-z0-9]", "", s.lower())


def foreign(s):
    if CJK.search(s):
        return True
    h, l = len(HANGUL.findall(s)), len(LATIN.findall(s))
    return h == 0 or h / (h + l) < 0.5


def truncated(raw):
    raw = str(raw).rstrip()
    return raw.endswith("...") and 200 <= len(raw) <= 206


def len_bucket(n):
    return pd.cut(pd.Series(n), BINS, labels=BIN_LABELS).astype(str).values


def build(norm_level=2, filters=True, dedupe=True, length_match=True, log=print):
    """norm_level 0: NFC만(1차와 동일) / 1: 줄바꿈·공백 / 2: 전체 정규화"""
    kr3 = load_kr3("kr3_raw.tsv")
    g1 = load_ai("generated.jsonl")
    g2 = load_ai("generated_clean_v2.jsonl")
    log(f"원본: KR3 {len(kr3):,} / gen1 {len(g1):,} / gen2 {len(g2):,}")

    kr3["raw"] = kr3.text
    for d in (kr3, g1, g2):
        d["text"] = d.text.map(lambda s: normalize(s, norm_level))

    if filters:
        m = kr3.raw.map(truncated); log(f"F1 잘린 사람 리뷰 제거: {m.sum():,}"); kr3 = kr3[~m]
        for name, d in (("KR3", kr3), ("gen1", g1), ("gen2", g2)):
            m = d.text.map(foreign) | (d.text.str.len() < 2)
            log(f"F2·F3 외국어/2자 미만 제거 {name}: {m.sum():,}")
        kr3 = kr3[~(kr3.text.map(foreign) | (kr3.text.str.len() < 2))]
        g1 = g1[~(g1.text.map(foreign) | (g1.text.str.len() < 2))]
        g2 = g2[~(g2.text.map(foreign) | (g2.text.str.len() < 2))]
    else:
        kr3 = kr3[kr3.text.str.len() > 0]

    if dedupe:
        for name in ("kr3", "g1", "g2"):
            d = locals()[name]; d = d.assign(k=d.text.map(key))
            n0 = len(d); d = d.drop_duplicates("text"); n1 = len(d); d = d[d.k.str.len() > 0].drop_duplicates("k")
            log(f"D1·D2 중복 제거 {name}: 완전 {n0-n1:,} / 준중복 {n1-len(d):,}")
            if name == "kr3": kr3 = d
            elif name == "g1": g1 = d
            else: g2 = d
        hk = set(kr3.k)
        for name in ("g1", "g2"):
            d = locals()[name]; m = d.k.isin(hk)
            log(f"D3 KR3와 겹치는 AI 글 제거 {name}: {m.sum():,}")
            if name == "g1": g1 = d[~m]
            else: g2 = d[~m]
    else:
        kr3 = kr3.drop_duplicates("text"); g1 = g1.drop_duplicates("text"); g2 = g2.drop_duplicates("text")

    if length_match:
        cap = int(g1.text.str.len().max())
        n0 = len(kr3); kr3 = kr3[kr3.text.str.len() <= cap]
        log(f"S1 AI 최대 길이({cap}자)보다 긴 사람 리뷰 제외: {n0-len(kr3):,}")
        ab = pd.Series(len_bucket(g1.text.str.len())).value_counts(normalize=True)
        hb = len_bucket(kr3.text.str.len())
        parts = []
        for b in BIN_LABELS:
            n = int(round(N_HUMAN * ab.get(b, 0)))
            parts.append(kr3[hb == b].sample(n, random_state=SEED))
        human = pd.concat(parts)
        log("S1 사람 길이 맞춤 추출: " + ", ".join(f"{b} {int(round(N_HUMAN*ab.get(b,0)))}" for b in BIN_LABELS))
    else:
        human = kr3.sample(N_HUMAN, random_state=SEED)

    human = pd.DataFrame({"text": human.text.values, "label": 0, "generator": "human", "prompt_type": "-",
                          "rating": human.Rating.values})
    def ai_frame(d):
        return pd.DataFrame({"text": d.text.values, "label": 1, "generator": d.generator.values,
                             "prompt_type": d.prompt_type.values, "rating": "-"})
    A1, A2 = ai_frame(g1), ai_frame(g2)
    for d in (human, A1, A2):
        d["len"] = d.text.str.len(); d["len_bucket"] = len_bucket(d["len"])

    def split(d, cols):
        strat = d[cols].astype(str).agg("_".join, axis=1)
        vc = strat.value_counts(); strat = strat.where(strat.map(vc) >= 2, "rare")
        tr, te = train_test_split(d.index, test_size=0.2, random_state=SEED, stratify=strat)
        d = d.copy(); d["split"] = "train"; d.loc[te, "split"] = "test"; return d

    human = split(human, ["len_bucket"])
    A1 = split(A1, ["len_bucket", "generator", "prompt_type"])
    A2 = split(A2, ["len_bucket", "generator", "prompt_type"])
    main = pd.concat([human, A1], ignore_index=True).assign(set="main")
    hold = A2.assign(set="holdout_gen2")
    log(f"최종: 사람 {len(human):,} / gen1 {len(A1):,} / gen2 {len(A2):,}")
    return pd.concat([main, hold], ignore_index=True)


if __name__ == "__main__":
    lines = []
    def log(s): print(s); lines.append(s)
    df = build(log=log)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/exp2_dataset.csv", index=False)
    m = df[df.set == "main"]
    log("\nmain 분할:\n" + pd.crosstab(m.split, m.label, margins=True).to_string())
    log("\n정제 후 표면 특징:\n" + pd.DataFrame({
        "사람": surface_profile(m[m.label == 0].text),
        "gen1": surface_profile(m[m.label == 1].text),
        "gen2": surface_profile(df[df.set == "holdout_gen2"].text)}).round(1).to_string())
    open(f"{OUT_DIR}/exp2_clean_log.txt", "w").write("\n".join(lines))
