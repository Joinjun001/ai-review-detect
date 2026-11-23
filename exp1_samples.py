"""1차 실험: 사람/AI 리뷰 예시 출력."""
import pandas as pd
from common import *
kr3 = load_kr3("kr3_raw.tsv")
g1 = load_ai("generated.jsonl"); g2 = load_ai("generated_clean_v2.jsonl")
def show(t, title, k=4, seed=7):
    print(f"\n### {title}")
    for s in t.sample(k, random_state=seed):
        s = str(s).replace("\n", " / "); print("-", s[:220] + ("…" if len(s) > 220 else ""))
show(kr3.text, "사람 KR3 원본", 6)
for gen in ["gpt", "claude"]:
    for p in ["A", "C"]:
        show(g1[(g1.generator == gen) & (g1.prompt_type == p)].text, f"gen1 {gen} 프롬프트 {p}")
for gen in ["gemma3", "gemma4", "qwen3"]:
    for p in ["A", "C"]:
        show(g2[(g2.generator == gen) & (g2.prompt_type == p)].text, f"gen2 {gen} 프롬프트 {p}", 3)
