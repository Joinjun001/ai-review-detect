"""1차 실험: 정제 없이 사람(KR3) 5,000 vs AI 리뷰를 그대로 넣었을 때 얼마나 쉽게 구별되는가.

사용법:
  python exp1_baseline.py kr3.tsv       # 전처리본
  python exp1_baseline.py kr3_raw.tsv   # 원본
"""
import sys
import pandas as pd
from common import *

kr3_name = sys.argv[1] if len(sys.argv) > 1 else "kr3_raw.tsv"
tag = "pre" if kr3_name == "kr3.tsv" else "raw"

kr3 = load_kr3(kr3_name)
kr3["text"] = kr3.text.map(nfc)
g1 = load_ai("generated.jsonl"); g1["text"] = g1.text.map(nfc)
g2 = load_ai("generated_clean_v2.jsonl"); g2["text"] = g2.text.map(nfc)

log = []
def p(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

p(f"# 1차 실험 — 사람: {kr3_name} ({len(kr3):,}건)")
kr3set = set(kr3.text)
p("KR3 리뷰와 글자까지 같은 AI 글:", "gen1", int(g1.text.isin(kr3set).sum()), "/ gen2", int(g2.text.isin(kr3set).sum()))
p("AI 글 내부 완전 중복:", "gen1", int(g1.text.duplicated().sum()), "/ gen2", int(g2.text.duplicated().sum()))

human = kr3[kr3.text.str.len() > 0].drop_duplicates("text").sample(5000, random_state=SEED).reset_index(drop=True)
A1 = g1.drop_duplicates("text").reset_index(drop=True)
A2 = g2.drop_duplicates("text").reset_index(drop=True)

prof = pd.DataFrame({
    "사람": surface_profile(human.text),
    "gen1 GPT/Claude": surface_profile(A1.text),
    "gen1 A": surface_profile(A1[A1.prompt_type == "A"].text),
    "gen1 C": surface_profile(A1[A1.prompt_type == "C"].text),
    "gen2 Gemma/Qwen": surface_profile(A2.text),
}).round(1)
p("\n## 표면 특징 (% / 글자 수)\n" + prof.to_string())
prof.to_csv(f"{OUT_DIR}/exp1_{tag}_surface.csv")

res = []
for t, kw in [
    ("gen1 같은 분포", dict(ai_train=A1)),
    ("gen2 같은 분포", dict(ai_train=A2)),
    ("gen1 학습 → gen2 평가", dict(ai_train=A1, ai_test=A2)),
    ("gen2 학습 → gen1 평가", dict(ai_train=A2, ai_test=A1)),
    ("gen1 A 학습 → C 평가", dict(ai_train=A1[A1.prompt_type == "A"], ai_test=A1[A1.prompt_type == "C"])),
]:
    r, _, _ = run_scenario(human, tag=t, **kw); res.append(r)

# 사람 길이 분포를 AI(gen1)에 맞춘 경우
bins = [0, 20, 50, 150, 10**6]
ab = pd.cut(A1.text.str.len(), bins).value_counts(normalize=True)
hb = pd.cut(kr3.text.str.len(), bins)
humL = pd.concat([kr3[hb == b].drop_duplicates("text").sample(int(round(5000 * f)), random_state=SEED) for b, f in ab.items()]).reset_index(drop=True)
r, _, _ = run_scenario(humL, ai_train=A1, tag="gen1 같은 분포, 사람 길이 맞춤"); res.append(r)

res = pd.concat(res, ignore_index=True)
res.to_csv(f"{OUT_DIR}/exp1_{tag}_results.csv", index=False)
p("\n## 결과 (0.5 기준)\n" + fmt(res).to_string(index=False))
open(f"{OUT_DIR}/exp1_{tag}_log.txt", "w").write("\n".join(log))
