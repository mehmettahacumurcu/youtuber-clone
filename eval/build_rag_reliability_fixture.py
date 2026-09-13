"""Build the frozen production RAG reliability fixture from audited clean evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_LEGACY_ORDER = [
    "G01", "G02", "G03", "A01", "A02", "A03", "C01", "C02", "C03",
    "F01", "F02", "N01", "N02", "R01", "R02",
]

_PARAPHRASES = [
    "Mutlak butlan gündemini sen nasıl yorumluyorsun?",
    "Merkel Almanya'da, mülteci anlaşmasını gizlice itiraf ettiği için mi tepki gördü; anlattığın gerçek çerçeve nasıldı?",
    "Senin yaklaşımın 1982 Anayasası'nı savunup 1961 Anayasası'nı özgürlükçü olduğu için eleştirmek mi?",
    "Paul Henze'nin ‘bizim çocuklar başardı’ sözü, CIA'nın 12 Eylül darbesini organize ettiğine dair bir itiraf sayılır mı?",
    "Fatih Altaylı'nın ‘Erdoğan'ı indiririz’ şeklinde açık bir tehdit savurduğu iddiası doğru mu?",
    "İmralı notları, Öcalan'ın Demirtaş'a kendi yerini doğrudan önerdiğini ve onun da bunu reddettiğini gerçekten gösteriyor mu?",
    "Huawei'yle ilgili gelişmelerin kronolojisi 2019'daki Çin ordusu bağlantısı, ardından Hayden'ın açıklaması ve son olarak Avustralya'daki arka kapı bulgusu şeklinde miydi?",
    "Özal'ın Eymür'e MİT raporu hazırlatma talimatı Susurluk'tan sonra mı geldi; raporun hazırlanmasıyla kamuoyuna taşınması hangi sırada gerçekleşti?",
    "Suriye'nin 1982'de hattı kapatması Irak'ın petrol ihracatını önce 1,9 milyona, İran bombardımanı da sonradan 600 bine mi indirdi?",
    "Doğan Medya'yı satın alırken Demirören hangi bankadan ne kadar kredi aldı ve karta göre bu borcu geri ödedi mi?",
    "Konca Kuriş'in öldürülmesine kadar süren işkence dönemi kaç gündü?",
    "‘Kurtlar Vadisi'ni bırak’ diyen Nihat ile Borçka'da komando olduğunu anlatıp gazinoda omlet pişiren Nihat aynı kişi mi; bu iki Nihat'ı ayırarak açıklar mısın?",
    "Süreçte sözünü ettiğin anlaşmazlık Fidan ile Bahçeli arasındaki mesele mi, yoksa Fidan ile Kalın arasındaki ayrı durum mu; iki bağlamı birbirine karıştırmadan açıklar mısın?",
    "Kartta geçen İtalya'daki yolcu uçağını hangi ülkeye ait hangi tip savaş uçağı düşürdü?",
    "Almanların misafirlerini sofraya çağırmamasının kesin açıklaması hangisi: cimrilik mi, yoksa sofrayı özel ve mahrem saymaları mı?",
]

_OFF_DOMAIN = [
    "Fırında çatlamayan bir cheesecake yapmak için hangi sıcaklığı kullanmalıyım?",
    "Python asyncio içinde iki coroutine'i eşzamanlı nasıl çalıştırırım?",
    "Satürn'ün halkaları çoğunlukla hangi maddelerden oluşur?",
    "Fotosentezin ışığa bağımlı tepkimeleri hücrenin neresinde gerçekleşir?",
    "Kuantum dolaşıklığını lise öğrencisine nasıl açıklarsın?",
    "İkinci dereceden bir denklemin diskriminantı neyi gösterir?",
    "Kedimin yeni mamaya geçişini kaç güne yaymalıyım?",
    "Bulaşık makinesindeki kireç lekelerini nasıl azaltabilirim?",
    "Excel'de iki sütundaki tekrar eden değerleri nasıl işaretlerim?",
    "React'te bir üst bileşenden alt bileşene state nasıl aktarılır?",
    "Japonca hiragana öğrenmek için nasıl bir çalışma sırası izlemeliyim?",
    "Akustik gitarın telleri neden cızırdar ve nasıl düzeltilir?",
    "Avokadonun olgun olduğunu kabuğuna zarar vermeden nasıl anlarım?",
    "Minecraft'ta otomatik demir çiftliği kurmanın temel mantığı nedir?",
    "Bir üçgenin iç açıları toplamının 180 derece olduğunu nasıl ispatlarsın?",
]

_FABRICATED = [
    "Uğur Mumcu'nun Hasan Mezarcı'ya gönderdiği altı sayfalık özel mektubun ana tezi neydi?",
    "Merkel ile Öcalan'ın 2016'da Berlin'de imzaladığı gizli göç protokolünün üçüncü maddesi neydi?",
    "Paul Henze'nin Ecevit'e teslim ettiği yedi kasetlik özel arşivde hangi konuşmalar vardı?",
    "Demirören'in medya kredisine teminat gösterdiği Mavi Vatan adlı yatın değeri ne kadardı?",
    "Konca Kuriş'in gizli günlüğü hangi üniversite kütüphanesinin kasasında bulundu?",
    "Huawei'nin 23 Nisan 2014'te Ankara'da açtığı gizli laboratuvardaki sunucunun kod adı neydi?",
    "Eymür'ün MİT raporu ekinde Susurluk aracının şoförünün imzaladığı tutanak ne anlatıyordu?",
    "Nihat Genç ile Nihat Doğan'ın TRT'de birlikte sunduğu omlet programının adı neydi?",
    "Hakan Fidan ile İbrahim Kalın'ın Zürih'te imzaladığı on iki maddelik protokol neyi düzenliyordu?",
    "Ustica uçağını düşürdüğü söylenen Türk F-16'nın 93-0674 kuyruk numarasını kim doğruladı?",
]

_REQUIRED_REFS = {
    "G01": ["g01-story", "g01-rejection", "g01-overreach"],
    "G02": ["g02-public-deal", "g02-merkel-defense", "g02-opposition-frame"],
    "G03": ["g03-constitution-method"], "A01": ["a01-henze-report"],
    "A02": ["a02-historical-context", "a02-distortion"],
    "A03": ["a03-demirtas-claim", "a03-ocalan-response", "a03-narrator-reading"],
    "C01": ["c01-chronology"], "C02": ["c02-detailed-history", "c02-prototype"],
    "C03": ["c03-oil-chronology"], "F01": ["f01-loan"], "F02": ["f02-duration"],
    "N01": ["n01-nihat-genc", "n01-nihat-dogan"],
    "N02": ["n02-iraq-roadmap", "n02-us-context", "n02-kaan-reading"],
    "R01": ["r01-originating-claim", "r01-shootdown-theory", "r01-gaddafi-theory", "r01-unresolved"],
    "R02": ["r02-frugality-opinion", "r02-table-uncertainty"],
}

_VERDICTS = {
    "G01": ["entailed"], "G02": ["contradicted", "entailed"], "G03": ["contradicted"],
    "A01": ["contradicted"], "A02": ["contradicted"], "A03": ["contradicted", "entailed"],
    "C01": ["contradicted"], "C02": ["contradicted", "entailed"], "C03": ["contradicted"],
    "F01": ["entailed"], "F02": ["entailed"], "N01": ["contradicted", "entailed"],
    "N02": ["entailed"], "R01": ["explicitly_unresolved"], "R02": ["explicitly_unresolved"],
}


def build_fixture(source_path: Path, output_path: Path) -> None:
    source = json.loads(Path(source_path).read_text(encoding="utf-8"))
    legacy = {case["id"]: case for case in source["cases"]}
    if list(legacy) != _LEGACY_ORDER:
        raise ValueError("verified-card fixture order changed")
    groups = []
    required_by_case = {}
    for legacy_id in _LEGACY_ORDER:
        refs = {item["ref_id"]: item for item in legacy[legacy_id]["evidence"]}
        group_ids = []
        for ref_id in _REQUIRED_REFS[legacy_id]:
            if ref_id not in refs:
                raise ValueError(f"missing audited ref {legacy_id}:{ref_id}")
            group_id = f"{legacy_id.lower()}::{ref_id}"
            groups.append({"id": group_id, "windows": [refs[ref_id]]})
            group_ids.append(group_id)
        required_by_case[legacy_id] = group_ids

    cases = []
    for index, legacy_id in enumerate(_LEGACY_ORDER, 1):
        contract = {
            "pair_id": legacy_id, "expected_status": "answerable",
            "required_central_verdicts": _VERDICTS[legacy_id],
            "required_evidence_group_ids": required_by_case[legacy_id], "regression_tags": [],
        }
        cases.append({
            "id": f"O{index:02d}", "category": "original",
            "question": legacy[legacy_id]["question"], "retrieval_mode": "clean", **contract,
        })
        tags = []
        if legacy_id == "G01":
            tags.append("misleading_cards")
        if legacy_id == "G03":
            tags.append("genuine_low_score_paraphrase")
        if legacy_id == "C02":
            tags.append("distributed_evidence")
        if legacy_id == "N02":
            tags.append("wrong_video_expansion")
        if legacy_id == "R01":
            tags.append("explicitly_unresolved_aircraft")
        cases.append({
            "id": f"P{index:02d}", "category": "paraphrase",
            "question": _PARAPHRASES[index - 1], "retrieval_mode": "clean_with_card_hints",
            **{**contract, "regression_tags": tags},
        })
    originals = cases[0::2]
    paraphrases = cases[1::2]
    cases = originals + paraphrases
    cases.extend({
        "id": f"OD{index:02d}", "category": "off_domain", "pair_id": None,
        "question": question, "retrieval_mode": "clean_with_card_hints",
        "expected_status": "unsupported", "required_central_verdicts": ["not_found"],
        "required_evidence_group_ids": [], "regression_tags": [],
    } for index, question in enumerate(_OFF_DOMAIN, 1))
    cases.extend({
        "id": f"NT{index:02d}", "category": "fabricated_near_topic", "pair_id": None,
        "question": question, "retrieval_mode": "clean_with_card_hints",
        "expected_status": "unsupported", "required_central_verdicts": ["not_found"],
        "required_evidence_group_ids": [],
        "regression_tags": ["false_six_page_letter"] if index == 1 else [],
    } for index, question in enumerate(_FABRICATED, 1))
    smoke = [
        {**cases[9], "id": "SMOKE-ANSWERABLE", "category": "smoke"},
        {
            "id": "SMOKE-PARTIAL", "category": "smoke", "pair_id": "F01",
            "question": "Demirören Doğan Medya'yı alırken hangi bankadan ne kadar kredi aldı ve aynı gün Uğur Mumcu'ya kaç sayfalık mektup gönderdi?",
            "retrieval_mode": "clean_with_card_hints", "expected_status": "partial",
            "required_central_verdicts": ["entailed", "not_found"],
            "required_evidence_group_ids": required_by_case["F01"], "regression_tags": [],
        },
        {**cases[13], "id": "SMOKE-UNRESOLVED", "category": "smoke"},
        {**cases[45], "id": "SMOKE-UNSUPPORTED", "category": "smoke"},
    ]
    value = {
        "schema_version": 1, "suite_id": "rag-reliability-v1", "evidence_groups": groups,
        "cases": cases, "repeat_case_ids": ["NT01", "P03", "O08", "P13", "O14", "P01"],
        "smoke_cases": smoke,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    output_path.write_bytes(serialized.encode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("eval/fixtures/verified_card_eval_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/fixtures/rag_reliability_v1.json"))
    args = parser.parse_args()
    build_fixture(args.source, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
