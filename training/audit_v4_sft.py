"""Throwaway audit of voice_sft_v4_{train,val}.jsonl — checks the rows AS WRITTEN."""
import json
import re
import sys

sys.path.insert(0, "E:/youtuber-clone")
from training.retrofit_v4_grounded import _content_stems, _numbers, _propers
from rag.prompt import SYS, SPAN_CHAR_CAP

DD = "E:/youtuber-clone/data/dataset"
HEADER = ("Aşağıda bu konu hakkında DAHA ÖNCE KENDİ söylediklerin var. Bunlara dayanarak, "
          "öğrendiğin anlatım üslubunu koruyarak soruyu cevapla. Uydurma; bu sözlerdeki "
          "görüşü kendi ağzından akıcı şekilde anlat.")


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


train = load(f"{DD}/voice_sft_v4_train.jsonl")
val = load(f"{DD}/voice_sft_v4_val.jsonl")
problems = []

# 1. structure + single SYS
sys_set = set()
for i, r in enumerate(train + val):
    m = r.get("messages")
    if not (isinstance(m, list) and len(m) == 3 and [x["role"] for x in m] == ["system", "user", "assistant"]
            and all(isinstance(x["content"], str) and x["content"].strip() for x in m)):
        problems.append(f"STRUCTURE row {i}")
    sys_set.add(m[0]["content"])
print(f"structure: {len(train)} train + {len(val)} val rows | SYS identical: {sys_set == {SYS}}")

# 2. duplicates + train/val leakage
key = lambda r: (r["messages"][1]["content"], r["messages"][2]["content"])
tk = [key(r) for r in train]
dups = len(tk) - len(set(tk))
leak = len(set(tk) & {key(r) for r in val})
print(f"exact-duplicate train rows: {dups} | train∩val leakage: {leak}")
if dups: problems.append(f"{dups} dup rows")
if leak: problems.append(f"{leak} leaked rows")

# 3. grounded rows: template shape + span constraints + SUPPORT AS-WRITTEN
g_rows = [r for r in train + val if "[Senin sözlerin]" in r["messages"][1]["content"]]
bad_tmpl = bad_span = num_fail = prop_fail = 0
num_fail_ex, prop_fail_ex = [], []
for r in g_rows:
    u, a = r["messages"][1]["content"], r["messages"][2]["content"]
    if not u.startswith(HEADER) or "[Soru]" not in u:
        bad_tmpl += 1
        continue
    block = u.split("[Senin sözlerin]\n", 1)[1].rsplit("\n\n[Soru]", 1)[0]
    spans = [s for s in block.split("\n- ") if s.strip()]
    if spans and spans[0].startswith("- "):
        spans[0] = spans[0][2:]
    if not (1 <= len(spans) <= 3) or any(len(s) > SPAN_CHAR_CAP + 20 for s in spans):
        bad_span += 1
    miss = _numbers(a) - _numbers(block) - _numbers(u.rsplit("[Soru]", 1)[1])  # question-echo ok
    if miss:
        num_fail += 1
        if len(num_fail_ex) < 3:
            num_fail_ex.append((sorted(miss), u.rsplit("[Soru]", 1)[1].strip()[:60]))
    props = _propers(a)
    if props:
        cov = len(props & _content_stems(block)) / len(props)
        if cov < 0.70:
            prop_fail += 1
            if len(prop_fail_ex) < 3:
                prop_fail_ex.append((round(cov, 2), u.rsplit("[Soru]", 1)[1].strip()[:60]))
print(f"grounded rows: {len(g_rows)} | bad template: {bad_tmpl} | bad span count/len: {bad_span}")
print(f"  SUPPORT as-written -> number-unsupported: {num_fail} | proper<0.70: {prop_fail}")
for ex in num_fail_ex: print("   num-fail:", ex)
for ex in prop_fail_ex: print("   prop-fail:", ex)
if bad_tmpl or bad_span: problems.append("template/span issues")
if num_fail or prop_fail: problems.append(f"{num_fail}+{prop_fail} support failures")

# 4. abstain rows present + bare format
ab_qs = {json.loads(l)["messages"][1]["content"] for l in open(f"{DD}/abstain_v4.jsonl", encoding="utf-8") if l.strip()}
ab_in = [r for r in train + val if r["messages"][1]["content"] in ab_qs]
bare = all("[Senin sözlerin]" not in r["messages"][1]["content"] for r in ab_in)
print(f"abstain rows in train+val: {len(ab_in)}/83 | all bare-format: {bare}")

# 5. length safety (MAX_SEQ_LEN=4096; repo est: chars*0.292)
mx = max(sum(len(m["content"]) for m in r["messages"]) for r in train)
print(f"longest row: {mx} chars ≈ {int(mx*0.292)} est tokens (limit 4096, truncation_side=left)")

print("\nVERDICT:", "CLEAN" if not problems else f"PROBLEMS: {problems}")
