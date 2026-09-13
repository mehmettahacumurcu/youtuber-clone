"""Stage A of the broad ASR-garble sweep: heuristic ranker over the WHOLE corpus.

No LLM. Scores every segment for garble-likelihood using unsupervised signals that survive Turkish
agglutination (a char-trigram language model trained on the corpus itself, so rare-but-legit
inflected forms score LOW while genuinely anomalous tokens score HIGH), combined with token rarity,
merge-length, foreign letters, consonant clusters, and Whisper's own avg_logprob.

This is HIGH-RECALL / modest-precision by design — the LLM pass (Stage B) and the human audio tool
are the precision filters. It catches NON-WORD garbles (Besozat, duayenak, Mutanika) well; it will
miss valid-word-substituted-in-context errors (sayfası->tayfası) — those need the LLM/ear.

Output: data/corrections/candidates.jsonl, ranked desc by score, each with the suspicious token(s)
so Stage B and the audio tool can target them. Prints the score distribution + a top sample.

Run:  .venv\\Scripts\\python.exe -m tools.detect_garbles [--top 3000]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter

from pipeline.config import load_settings

WORD_RE = re.compile(r"[0-9A-Za-zÇĞİıÖŞÜçğöşüâîûÂÎÛ'’]+")
TR_VOWELS = set("aeıioöuüâîûAEIİOÖUÜ")
FOREIGN = set("qwxQWX")
COMMON_FOREIGN_OK = {"twitter", "youtube", "whatsapp", "wifi", "wikipedia", "max", "marx", "wagner"}


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str) -> list[str]:
    return [w.strip("'’") for w in WORD_RE.findall(text) if w.strip("'’")]


def consonant_run(tok: str) -> int:
    best = run = 0
    for ch in tok:
        if ch.isalpha() and ch not in TR_VOWELS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def build_trigram_model(all_tokens: Counter) -> tuple[Counter, int]:
    tri = Counter()
    for tok, c in all_tokens.items():
        s = "^" + tok + "$"
        for i in range(len(s) - 2):
            tri[s[i:i + 3]] += c
    return tri, sum(tri.values())


def token_nll(tok: str, tri: Counter, total: int, vocab: int) -> float:
    """Mean negative log-prob per trigram (add-1 smoothed). High = anomalous for this corpus."""
    s = "^" + tok + "$"
    grams = [s[i:i + 3] for i in range(len(s) - 2)]
    if not grams:
        return 0.0
    return sum(-math.log((tri.get(g, 0) + 1) / (total + vocab)) for g in grams) / len(grams)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3000)
    ap.add_argument("--rare-max", type=int, default=4, help="token corpus freq <= this to be a candidate")
    args = ap.parse_args()

    s = load_settings()
    clean_dir = s.paths.clean_dir

    # pass 1: load segments + global token frequency (lowercased)
    segs: list[dict] = []
    freq: Counter = Counter()
    for fp in sorted(clean_dir.glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        vid = doc.get("video_id") or fp.stem
        title = doc.get("title") or ""
        for i, seg in enumerate(doc.get("segments", [])):
            text = seg.get("text") or ""
            toks = tokenize(text)
            for t in toks:
                freq[tr_lower(t)] += 1
            segs.append({"video_id": vid, "title": title, "seg_index": i,
                         "start": seg.get("start"), "end": seg.get("end"),
                         "text": text, "avg_logprob": seg.get("avg_logprob"), "_toks": toks})

    tri, total = build_trigram_model(freq)
    vocab = len(tri)
    # corpus NLL stats to set an anomaly threshold (mean+std over types, freq-weighted-ish)
    nlls = {tok: token_nll(tok, tri, total, vocab) for tok in freq}
    vals = sorted(nlls.values())
    p_hi = vals[int(0.85 * len(vals))]  # 85th percentile NLL = "anomalous"

    def suspicious(tok: str) -> tuple[bool, float]:
        low = tr_lower(tok)
        if not tok[0].isalpha() or len(low) < 5:
            return False, 0.0
        if low in COMMON_FOREIGN_OK:
            return False, 0.0
        f = freq.get(low, 0)
        if f > args.rare_max:
            return False, 0.0
        nll = nlls.get(low, 0.0)
        cap = tok[0].isupper()
        foreign = any(c in FOREIGN for c in tok)
        crun = consonant_run(low)
        novowel = not any(c in TR_VOWELS for c in low)
        merge = len(low) >= 14
        anomaly = (nll >= p_hi) or foreign or crun >= 4 or novowel or merge
        if not anomaly and not cap:
            return False, 0.0
        # score the token
        score = 0.0
        score += max(0.0, nll - p_hi) * 2.0
        score += (5 - f) * 0.4                 # rarer = higher
        score += 1.5 if foreign else 0.0
        score += 1.0 * max(0, crun - 3)
        score += 2.0 if novowel else 0.0
        score += 1.5 if merge else 0.0
        score += 0.6 if cap else 0.0
        return True, score

    COMMON = 50  # corpus freq for a token to count as "common Turkish"

    def is_common_tr(tok: str) -> bool:
        return freq.get(tr_lower(tok), 0) >= COMMON and not any(c in FOREIGN for c in tok)

    def ascii_alpha(tok: str) -> bool:
        return len(tok) >= 3 and all("a" <= c.lower() <= "z" for c in tok if c.isalpha()) and tok.isascii()

    cands, foreign_passages = [], 0
    for seg in segs:
        toks = seg["_toks"]
        n_alpha = sum(1 for t in toks if t[:1].isalpha())
        if not n_alpha:
            continue
        # drop foreign/English read-aloud passages (a RUN of foreign tokens, not a lone garble)
        foreignness = sum(1 for t in toks if ascii_alpha(t)) / n_alpha
        if foreignness >= 0.40:
            foreign_passages += 1
            continue
        flagged = []
        sc = 0.0
        for p, t in enumerate(toks):
            ok, ts = suspicious(t)
            if not ok:
                continue
            # ISOLATION: a real garble sits next to common Turkish; a foreign run does not
            neigh_common = any(0 <= q < len(toks) and is_common_tr(toks[q]) for q in (p - 1, p + 1))
            if not neigh_common:
                continue
            flagged.append(t)
            sc += ts
        if not flagged:
            continue
        k = len(flagged)
        sc *= 2.0 if k == 1 else (1.1 if k == 2 else 0.4)  # reward isolation, punish foreign-name lists
        lp = seg.get("avg_logprob")
        if isinstance(lp, (int, float)):
            sc += max(0.0, (-lp - 0.2)) * 3.0
        cands.append({"video_id": seg["video_id"], "title": seg["title"],
                      "seg_index": seg["seg_index"], "start": seg["start"], "end": seg["end"],
                      "text": seg["text"], "avg_logprob": seg["avg_logprob"],
                      "suspect_tokens": sorted(set(flagged)), "score": round(sc, 3)})
    print(f"foreign/read-aloud passages skipped: {foreign_passages}")

    cands.sort(key=lambda c: c["score"], reverse=True)
    out = s.paths.dataset_dir.parent / "corrections" / "candidates.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    top = cands[: args.top]
    out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in top) + "\n", encoding="utf-8")

    print(f"segments scanned : {len(segs)}")
    print(f"unique tokens    : {len(freq)}")
    print(f"candidate segs   : {len(cands)}  (writing top {len(top)} -> {out})")
    if cands:
        sc = [c["score"] for c in cands]
        print(f"score range      : {sc[-1]:.2f} .. {sc[0]:.2f}   median {sc[len(sc)//2]:.2f}")
        print("\n--- TOP 30 (score · suspect_tokens · text) ---")
        for c in cands[:30]:
            toks = ",".join(c["suspect_tokens"])
            print(f"{c['score']:6.2f}  [{toks}]  {c['text'][:90]}")


if __name__ == "__main__":
    main()
