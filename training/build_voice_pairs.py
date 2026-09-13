"""Phase-2 step 1: seed the voice-SFT dataset.

For each curated his-topic question, RAG-retrieve his REAL chunks. Keep only the questions that
ground well (top rerank score >= threshold), so the teacher (next step) writes answers anchored in
his actual words. Output: data/dataset/voice_seeds_retrieved.jsonl with {question, context, sources}.

Run:  .venv\\Scripts\\python.exe training/build_voice_pairs.py
(Needs the local RAG index at data/rag/qdrant + BGE-M3 cached.)
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import load_settings
from rag.prompt import _mmss
from rag.retrieve import retrieve

# Curated, diverse seed questions across his real recurring subjects (analytical + casual + specific
# registers). Retrieval filters to the ones his corpus actually covers, so answers stay grounded.
SEED_QUESTIONS = [
    # CHP / opposition
    "CHP hakkında ne düşünüyorsun?",
    "Kılıçdaroğlu nasıl bir siyasetçi sence?",
    "Şu '100 yıllık parti' lafı hakkında ne diyorsun?",
    "İmamoğlu'nun siyasi geleceğini nasıl görüyorsun?",
    "Özgür Özel'in CHP'yi nereye götürdüğünü düşünüyorsun?",
    "CHP'nin kemalizmle ilişkisi hakkında ne dersin?",
    "Ya bu CHP'liler neden hep aynı şeyi yapıyor?",
    "CHP ve AKP arasındaki ilişki hakkında ne düşünüyorsun?",
    "Sosyal demokratlık Türkiye'de ne durumda?",
    # AKP / Erdoğan
    "Erdoğan'ın siyaseti hakkında ne düşünüyorsun?",
    "AKP'nin iktidardaki dönüşümünü nasıl değerlendiriyorsun?",
    "Erdoğan ve Öcalan ilişkisi hakkında ne dersin?",
    "AKP'nin dış politikası sence nasıl?",
    # Öcalan / PKK / terör
    "Öcalan hakkında ne düşünüyorsun?",
    "PKK meselesini nasıl görüyorsun?",
    "Apo'nun son açıklamaları hakkında ne dersin?",
    "Çözüm süreci hakkında ne düşünüyorsun?",
    "Kürt meselesi hakkında ne diyorsun?",
    "Devletin PKK ile pazarlığı hakkında ne dersin?",
    # MHP / milliyetçilik
    "MHP ve Bahçeli hakkında ne düşünüyorsun?",
    "Türk milliyetçiliği bugün ne durumda sence?",
    "Ülkücü hareket hakkında ne dersin?",
    # Almancılar / diaspora
    "Almanya'daki Türkler hakkında ne düşünüyorsun?",
    "Gurbetçiler hakkında ne diyorsun?",
    "Avrupa'daki Türk diasporası nasıl bir yerde?",
    "Almanların Türklere bakışı hakkında ne dersin?",
    "Almanya'daki PKK faaliyetleri hakkında ne düşünüyorsun?",
    # Suriye / mülteci / dış
    "Suriyeli mülteciler meselesi hakkında ne diyorsun?",
    "Suriye iç savaşını nasıl değerlendiriyorsun?",
    "Türkiye'nin Suriye politikası sence nasıl?",
    # ABD / Batı / geopolitik
    "Amerika'nın Türkiye politikası hakkında ne düşünüyorsun?",
    "Trump hakkında ne dersin?",
    "NATO ve Türkiye ilişkisi hakkında ne diyorsun?",
    "Gladio ve derin devlet hakkında ne düşünüyorsun?",
    "Batı'nın Ortadoğu'daki oyunları hakkında ne dersin?",
    "İsrail meselesi hakkında ne düşünüyorsun?",
    # derin devlet / darbeler
    "15 Temmuz darbe girişimi hakkında ne düşünüyorsun?",
    "28 Şubat süreci hakkında ne dersin?",
    "80 darbesi hakkında ne diyorsun?",
    "Türkiye'de darbe geleneği hakkında ne düşünüyorsun?",
    "Askeri vesayet meselesini nasıl görüyorsun?",
    "Fetö meselesi hakkında ne dersin?",
    # tarih / Cumhuriyet
    "Atatürk ve Cumhuriyet hakkında ne düşünüyorsun?",
    "Özal dönemini nasıl değerlendiriyorsun?",
    "Osmanlı'nın çöküşü hakkında ne dersin?",
    "Türk tarihinde en kritik dönem sence hangisi?",
    "Cumhuriyet'in kuruluşu hakkında ne düşünüyorsun?",
    # medya / kültür / toplum
    "Türk medyası hakkında ne düşünüyorsun?",
    "Sosyal medyanın siyasete etkisi hakkında ne dersin?",
    "Türkiye'de gazetecilik ne durumda sence?",
    "Türk toplumunun siyasi bilinci hakkında ne diyorsun?",
    "Bu yorumcular, analistler hakkında ne düşünüyorsun?",
    "İnsanların seni neden takip ettiğini düşünüyorsun?",
    # casual / reaction register
    "Ya bu memleketin hali ne olacak böyle?",
    "Türkiye'nin geleceğini nasıl görüyorsun?",
    "Şu an siyasette en çok neye sinirleniyorsun?",
    "Muhalefetin hali hakkında ne dersin?",
    "Halk neden bu kadar kolay kandırılıyor sence?",
    "Komplo teorileri hakkında ne düşünüyorsun?",
    "Türkiye'de ekonomi neden böyle?",
    "Seçimler hakkında ne diyorsun?",
]


def main() -> None:
    s = load_settings()
    r = s.rag
    out_path = Path(s.paths.dataset_dir) / "voice_seeds_retrieved.jsonl"
    kept, dropped = 0, 0
    rows = []
    for q in SEED_QUESTIONS:
        hits = retrieve(q, store_path=str(r.store_path), collection=r.collection,
                        embedder_name=r.embedder, reranker_name=r.reranker,
                        top_k=r.retrieve_top_k, top_n=r.rerank_top_n)
        if not hits or hits[0].rerank_score < r.gate_threshold:
            dropped += 1
            print(f"  drop (ungrounded {hits[0].rerank_score if hits else None}): {q}")
            continue
        rows.append({
            "question": q,
            "context": [h.chunk.text.strip() for h in hits],
            "sources": [f'"{h.chunk.title}" @ {_mmss(h.chunk.start)}' for h in hits],
            "top_score": round(hits[0].rerank_score, 2),
        })
        kept += 1
        print(f"  keep ({hits[0].rerank_score:.2f}): {q}")
    out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                        encoding="utf-8")
    print(f"\nKept {kept} grounded seeds, dropped {dropped} -> {out_path}")


if __name__ == "__main__":
    main()
