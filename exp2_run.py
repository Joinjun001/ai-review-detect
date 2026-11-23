"""2차 실험: 정제한 데이터로 다시 평가.

 (1) 단계별 정제 효과: 정제를 하나씩 더할 때 같은 분포(gen1) 점수가 어떻게 바뀌나
 (2) 정제 최종본에서 일반화 시나리오 6개
 (3) 표기 제거 ablation: 낱자·문장부호·이모지·기호를 양쪽에서 지운 뒤 재평가
 (4) 길이 구간별 성능 (TF-IDF, gen1 같은 분포)
"""
import re
import pandas as pd
from common import *
from exp2_clean import build

lines = []
def log(s=""): print(s); lines.append(str(s))

def evaluate(tr, te, tag, models=("표면특징15", "TF-IDF")):
    rows, probs = [], {}
    for m in models:
        p = fit_predict(m, tr, te); probs[m] = p
        rows.append({"조건": tag, "모델": m, **metrics(te.y, p)})
    return pd.DataFrame(rows), probs

def tt(d):
    d = d.rename(columns={"label": "y"}) if "label" in d else d
    return d[d.split == "train"], d[d.split == "test"]

# ---------- (1) 단계별 ----------
log("## (1) 단계별 정제 효과 — gen1 같은 분포")
steps = [
    ("S0 정제 없음 (NFC만)", dict(norm_level=0, filters=False, dedupe=False, length_match=False)),
    ("S1 + 정규화 (줄바꿈·공백·따옴표·목록기호)", dict(norm_level=2, filters=False, dedupe=False, length_match=False)),
    ("S2 + 잘린 리뷰·외국어 제거", dict(norm_level=2, filters=True, dedupe=False, length_match=False)),
    ("S3 + 중복·준중복·교차 중복 제거", dict(norm_level=2, filters=True, dedupe=True, length_match=False)),
    ("S4 + 길이 분포 맞춤 (최종)", dict(norm_level=2, filters=True, dedupe=True, length_match=True)),
]
step_res = []
for name, kw in steps:
    d = build(log=lambda s: None, **kw)
    d = d[d.set == "main"].rename(columns={"label": "y"})
    tr, te = tt(d)
    r, _ = evaluate(tr, te, name); step_res.append(r)
step_res = pd.concat(step_res, ignore_index=True)
log(fmt(step_res).to_string(index=False))
step_res.to_csv(f"{OUT_DIR}/exp2_steps.csv", index=False)

# ---------- (2) 최종본 시나리오 ----------
df = pd.read_csv("data/exp2_dataset.csv").rename(columns={"label": "y"})
H = df[(df.set == "main") & (df.y == 0)]
G1 = df[(df.set == "main") & (df.y == 1)]
G2 = df[df.set == "holdout_gen2"]
Htr, Hte = tt(H); G1tr, G1te = tt(G1); G2tr, G2te = tt(G2)
cat = lambda *a: pd.concat(a, ignore_index=True)

scen = [
    ("gen1 같은 분포", cat(Htr, G1tr), cat(Hte, G1te)),
    ("gen2 같은 분포", cat(Htr, G2tr), cat(Hte, G2te)),
    ("gen1 학습 → gen2 평가", cat(Htr, G1tr), cat(Hte, G2)),
    ("gen2 학습 → gen1 평가", cat(Htr, G2tr), cat(Hte, G1)),
    ("gen1 A 학습 → C 평가", cat(Htr, G1tr[G1tr.prompt_type == "A"]), cat(Hte, G1[G1.prompt_type == "C"])),
    ("gen1 C 학습 → A 평가", cat(Htr, G1tr[G1tr.prompt_type == "C"]), cat(Hte, G1[G1.prompt_type == "A"])),
]
log("\n## (2) 정제 최종본 — 시나리오별")
res = []
for tag, tr, te in scen:
    r, pr = evaluate(tr, te, tag, models=("길이만", "표면특징15", "TF-IDF")); res.append(r)
    if tag == "gen1 같은 분포":
        main_te, main_p = te, pr["TF-IDF"]
res = pd.concat(res, ignore_index=True)
log(fmt(res).to_string(index=False))
res.to_csv(f"{OUT_DIR}/exp2_scenarios.csv", index=False)

# ---------- (3) 표기 제거 ablation ----------
STRIP = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]|[^\w\s]|_", re.UNICODE)
def strip_surface(d):
    d = d.copy(); d["text"] = d.text.map(lambda s: re.sub(r"\s+", " ", STRIP.sub(" ", s)).strip()); return d
log("\n## (3) 표기 제거 ablation (낱자·문장부호·이모지·기호를 양쪽에서 제거, 띄어쓰기·어휘는 유지)")
abl = []
for tag, tr, te in [scen[0], scen[2], scen[4]]:
    r, _ = evaluate(strip_surface(tr), strip_surface(te), tag + " [표기 제거]", models=("TF-IDF",)); abl.append(r)
abl = pd.concat(abl, ignore_index=True)
log(fmt(abl).to_string(index=False))
abl.to_csv(f"{OUT_DIR}/exp2_ablation_surface.csv", index=False)

# ---------- (4) 길이 구간별 ----------
log("\n## (4) 길이 구간별 — TF-IDF, gen1 같은 분포")
rows = []
for b in ["≤20", "21–50", "51–150", ">150"]:
    m = (main_te.len_bucket == b).values
    rows.append({"길이": b, **metrics(main_te.y.values[m], main_p[m])})
lb = pd.DataFrame(rows); log(fmt(lb).to_string(index=False))
lb.to_csv(f"{OUT_DIR}/exp2_length.csv", index=False)

# 오류 예시
te = main_te.assign(p=main_p)
log("\n## 놓친 AI 글 (점수 낮은 순)")
for _, r in te[(te.y == 1) & (te.p < .5)].sort_values("p").head(10).iterrows():
    log(f"- [{r.generator}/{r.prompt_type} {r.p:.2f}] {r.text[:120]}")
log("\n## AI로 오탐된 사람 글 (점수 높은 순)")
for _, r in te[(te.y == 0) & (te.p >= .5)].sort_values("p", ascending=False).head(10).iterrows():
    log(f"- [{r.p:.2f}] {r.text[:120]}")

open(f"{OUT_DIR}/exp2_log.txt", "w").write("\n".join(lines))
