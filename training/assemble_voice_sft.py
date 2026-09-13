"""Phase-2 step 2: assemble the voice-SFT chat dataset.

(b) For each teacher-written pair, RE-RETRIEVE RAG context for its question and build a chat sample
    [system, user(context+question), assistant(answer)] — so the training input has the SAME shape as
    inference (RAG context + question -> his answer); otherwise the substance gap returns.
(a) Add a light set of his REAL coherent spans as cadence seasoning.
Shuffle (seeded), split -> data/dataset/voice_sft_{train,val}.jsonl.

Run:  .venv\\Scripts\\python.exe -m training.assemble_voice_sft
"""
from __future__ import annotations

import json
import random

from pipeline.config import load_settings
from rag.retrieve import retrieve

# Must match the SYSTEM prompt the inference path / notebook uses.
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "YouTuber'sin. Sana kendi videolarindan alintilar verilir; SADECE o alintilardaki bilgiyle, "
       "kendi uslubunla (senli benli, dogal kufurlu), akici ve anlasilir Turkce yanitla.")

A_PROMPTS = [
    "Bu konuda ne dusunuyorsun, kendi uslubunla anlat.",
    "Sunu bir acsana, ne diyorsun bu mevzuya?",
    "Bunu nasil degerlendiriyorsun abi?",
    "Bu konuyu biraz acalim, sen ne dersin?",
]


def _chat(user: str, assistant: str) -> dict:
    return {"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


def b_examples(s) -> list[dict]:
    r = s.rag
    # merge every generation pass (voice_pairs_raw*.jsonl) and dedup by (question, answer)
    pairs, seen = [], set()
    for fp in sorted(s.paths.dataset_dir.glob("voice_pairs_raw*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            key = (p["question"].strip(), p["answer"].strip()[:80])
            if key in seen:
                continue
            seen.add(key)
            pairs.append(p)
    print(f"  merged {len(pairs)} unique pairs from generation passes")
    out = []
    for i, p in enumerate(pairs):
        hits = retrieve(p["question"], store_path=str(r.store_path), collection=r.collection,
                        embedder_name=r.embedder, reranker_name=r.reranker,
                        top_k=r.retrieve_top_k, top_n=r.rerank_top_n)
        ctx = ("\n\n".join(h.chunk.text.strip() for h in hits) if hits
               else "\n\n".join(p.get("context", [])))
        out.append(_chat(f"Alintilar:\n{ctx}\n\nSoru: {p['question']}", p["answer"]))
        if (i + 1) % 10 == 0:
            print(f"  b {i+1}/{len(pairs)}")
    return out


def a_examples(s, n: int, rng: random.Random) -> list[dict]:
    spans = []
    for fp in sorted(s.paths.clean_dir.glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        doc = json.loads(fp.read_text(encoding="utf-8"))
        for seg in doc.get("segments", []):
            t = (seg.get("text") or "").strip()
            if len(t.split()) >= 30 and t and t[-1] in ".!?":
                spans.append(t)
    rng.shuffle(spans)
    return [_chat(rng.choice(A_PROMPTS), t) for t in spans[:n]]


def main() -> None:
    s = load_settings()
    rng = random.Random(7)
    b = b_examples(s)
    a = a_examples(s, n=24, rng=rng)
    allx = b + a
    rng.shuffle(allx)
    val_n = 8
    val, train = allx[:val_n], allx[val_n:]
    dd = s.paths.dataset_dir
    (dd / "voice_sft_train.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train) + "\n", encoding="utf-8")
    (dd / "voice_sft_val.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in val) + "\n", encoding="utf-8")
    print(f"\nDONE: {len(train)} train + {len(val)} val  ({len(b)} b-pairs + {len(a)} a-spans)")


if __name__ == "__main__":
    main()
