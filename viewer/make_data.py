"""데이터 뷰어용 샘플·통계 추출 → viewer/reviews.json"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from common import *
from exp2_clean import normalize, foreign, truncated

rng = 42
kr3 = load_kr3("kr3_raw.tsv")
g1 = load_ai("generated.jsonl"); g2 = load_ai("generated_clean_v2.jsonl")
ex2 = pd.read_csv("data/exp2_dataset.csv")

recs = []
def add(df, group, src, model, prompt, n, note=None, seed=rng):
    d = df.sample(min(n, len(df)), random_state=seed)
    for _, r in d.iterrows():
        raw = str(r["text"] if "raw" not in r else r["raw"])
        reason = None
        if src == "human":
            if truncated(raw): reason = "크롤링 잘림"
            elif foreign(normalize(raw)): reason = "외국어"
        recs.append({"g": group, "src": src, "model": model, "prompt": prompt, "raw": raw,
                     "clean": None if reason else normalize(raw), "drop": reason, "note": note})

kr3["raw"] = kr3.text
L = kr3.raw.str.len()
add(kr3, "human_raw", "human", "KR3 원본", "-", 40, seed=1)
add(kr3[kr3.raw.map(truncated)], "human_raw", "human", "KR3 원본", "-", 8, note="203자 잘림", seed=2)
add(kr3[kr3.raw.str.contains("\n") & (L < 200)], "human_raw", "human", "KR3 원본", "-", 8, note="줄바꿈", seed=3)
add(kr3[kr3.raw.map(lambda s: foreign(normalize(s))) & (L < 220)], "human_raw", "human", "KR3 원본", "-", 6, note="외국어", seed=4)

for gen, name in [("gpt", "GPT-4.1 mini"), ("claude", "Claude Haiku 4.5")]:
    for p in ["A", "C"]:
        add(g1[(g1.generator == gen) & (g1.prompt_type == p)], f"{gen}_{p}", "ai", name, p, 30)
for gen, name in [("gemma3", "Gemma3 12B"), ("gemma4", "Gemma4 12B"), ("qwen3", "Qwen3 8B")]:
    for p in ["A", "C"]:
        add(g2[(g2.generator == gen) & (g2.prompt_type == p)], f"{gen}_{p}", "ai", name, p, 20)

# 통계: 전체 데이터 기준
def prof(t):
    t = t.astype(str)
    P = {"jamo": r"[ㄱ-ㅎㅏ-ㅣ]", "emoji": r"[\U0001F300-\U0001FAFF☀-➿]", "nl": r"\n",
         "tilde": r"~", "ellip": r"\.\.|…", "digit": r"\d", "latin": r"[A-Za-z]",
         "formal": r"니다[.!~ ]*$", "yo": r"요[.!~^ ]*$", "period": r"\.$"}
    r = {k: round(t.str.contains(v, regex=True).mean() * 100, 1) for k, v in P.items()}
    r["trunc"] = round(t.map(truncated).mean() * 100, 1)
    r["len"] = round(t.str.len().median())
    r["n"] = int(len(t))
    return r
stats = {
    "사람 · 원본": prof(kr3.raw.sample(50000, random_state=rng)),
    "사람 · 정제 후": prof(ex2[(ex2.set == "main") & (ex2.label == 0)].text),
    "GPT · A": prof(g1[(g1.generator == "gpt") & (g1.prompt_type == "A")].text),
    "GPT · C": prof(g1[(g1.generator == "gpt") & (g1.prompt_type == "C")].text),
    "Claude · A": prof(g1[(g1.generator == "claude") & (g1.prompt_type == "A")].text),
    "Claude · C": prof(g1[(g1.generator == "claude") & (g1.prompt_type == "C")].text),
    "Gemma3": prof(g2[g2.generator == "gemma3"].text),
    "Gemma4": prof(g2[g2.generator == "gemma4"].text),
    "Qwen3": prof(g2[g2.generator == "qwen3"].text),
}
stats["사람 · 원본"]["n"] = int(len(kr3))
json.dump({"reviews": recs, "stats": stats}, open("viewer/reviews.json", "w"), ensure_ascii=False)
print(len(recs), "reviews,", os.path.getsize("viewer/reviews.json") // 1024, "KB")
print(json.dumps(stats, ensure_ascii=False)[:600])
