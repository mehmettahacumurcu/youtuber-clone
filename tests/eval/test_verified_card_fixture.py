from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from eval.card_eval_schema import load_suite, validate_suite


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "eval" / "fixtures" / "verified_card_eval_v1.json"

EXPECTED_SUITE_SHA256 = "cbdec61f4683853d5de5997d78c3c18b7a05be73509471de1328ed2b39b0b9c8"
EXPECTED_CASE_SHA256 = (
    ("G01", "473418b7de93a7626023be57bf799744a526c36bd566731dd270c6e2dab2056e"),
    ("G02", "9e9f31cd2ef07522436b22e178e426d49b1a1d8095b179c16eb9a9e5a3e24e00"),
    ("G03", "e472c910b4291ed723f0453244b092c9154bf9d6312244026b9be513603a299e"),
    ("A01", "d6336a59abdec412fa23b32925fa6ff818429f4ecbdccf2246e7e8fefe9ae7ef"),
    ("A02", "5596c39370c9a100a1f70e2d5db43a1d5bbe53815f90f2a5fd53a7daeca0cc30"),
    ("A03", "5b429b5688e0f693739fac8da259d8d212c457d64dedb3960c1217aad37482db"),
    ("C01", "982c56cd3464e8fc73b4fdd327aac7ffa184600c20c8c373f72fab0138fd5046"),
    ("C02", "edec291c61e327c588a8f7048327a918c1ae53bc106a5cbb43ee844b0abd71a3"),
    ("C03", "04327b7375e5b400129501df5473d48dbe7eb92e57ecd43e2238e7a65bc3ac74"),
    ("F01", "482987c77155637f0cccad6729a711a23919d0a02dfb971abde99137f29a28e0"),
    ("F02", "b5496870d2130e2fa9bdc57fd039344a374253b889f40e54671c5adbffd0137b"),
    ("N01", "6660258185b85a19a45e911c825facbf8ca694b71b37492aaac09d78093ebba5"),
    ("N02", "c9ac1800b26cbe0c646265d0a57a3804bea6856f3dd1cb6c0228168e807e6c77"),
    ("R01", "5cc5777694008fb1161d0398714131c050295b36e7b9f3323e2f9cb5c0def26c"),
    ("R02", "6d467cd4c293acf6940a1c4efe17c61b6ac0ad18730a0897f1ef2162243c7a5d"),
)

EXPECTED_CASES = [
    (
        "G01",
        "grounded",
        "Mutlak butlan meselesi hakkında ne düşünüyorsun?",
        ("card::2283", "card::2284", "card::2286"),
    ),
    (
        "G02",
        "grounded",
        "Merkel mülteci anlaşmasını gizlice itiraf ettiği için mi Almanya'da eleştirildi; senin anlattığın asıl tablo ne?",
        ("card::1107", "card::1109"),
    ),
    (
        "G03",
        "grounded",
        "Yani sen 82 Anayasası'nı savunup 61'i özgürlükçü olduğu için mi eleştiriyorsun?",
        ("card::0133", "card::1752"),
    ),
    (
        "A01",
        "grounded",
        "Paul Henze “bizim çocuklar başardı” diyerek CIA'nın 12 Eylül'ü yaptırdığını itiraf etmiş, doğru mu?",
        ("card::0251",),
    ),
    (
        "A02",
        "grounded",
        "Fatih Altaylı açıkça “Erdoğan'ı indiririz” diye tehdit etmişti, değil mi?",
        ("card::0568",),
    ),
    (
        "A03",
        "grounded",
        "Öcalan Demirtaş'a yerini açıkça teklif etmiş, Demirtaş da reddetmiş; İmralı notları bunu doğrulamıyor mu?",
        ("card::1521", "card::1522"),
    ),
    (
        "C01",
        "grounded",
        "Huawei kanıtları önce 2019'daki Çin ordusu bağlantısıyla çıkmış, sonra Hayden konuşmuş, en son Avustralya'daki arka kapı bulunmuştu; sıra böyle mi?",
        ("card::1357", "card::1976"),
    ),
    (
        "C02",
        "grounded",
        "Susurluk ortaya çıktıktan sonra mı Özal, Eymür'e MİT raporu hazırlattı; raporun oluşup kamuya taşınma sırası neydi?",
        ("card::0155", "card::0299"),
    ),
    (
        "C03",
        "grounded",
        "Irak petrol ihracatı önce Suriye 1982'de hattı kapatınca 1,9 milyona, sonra İran bombardımanıyla 600 bine düştü, doğru mu?",
        ("card::1307",),
    ),
    (
        "F01",
        "grounded",
        "Demirören Doğan medyasını alırken krediyi hangi bankadan, ne kadar çekti ve karta göre geri ödedi mi?",
        ("card::1170",),
    ),
    (
        "F02",
        "grounded",
        "Konca Kuriş öldürülmeden önce kaç gün işkence gördü?",
        ("card::2207",),
    ),
    (
        "N01",
        "grounded",
        "Sana “Kurtlar Vadisi işini bırak” diyen Nihat, Borçka'da komando olduğunu söyleyip gazinoda omlet yapan Nihat'la aynı kişi mi? İkisini karıştırmadan anlat.",
        ("card::1037", "card::1836", "card::1837"),
    ),
    (
        "N02",
        "grounded",
        "Süreçteki ayrışma Fidan–Bahçeli arasında mı, Fidan–Kalın arasında mı? Aynı kavga gibi anlatmadan iki bağlamı ayır.",
        ("card::0608", "card::2198"),
    ),
    (
        "R01",
        "abstain",
        "Kartta anlatılan İtalya'daki yolcu uçağı kazasında uçağı hangi ülkenin hangi savaş uçağı vurdu?",
        ("card::1339",),
    ),
    (
        "R02",
        "abstain",
        "Almanların misafiri sofraya çağırmamasının kesin nedeni ne: cimrilik mi, sofrayı mahrem görmeleri mi?",
        ("card::1082",),
    ),
]

CORRECTED_BINDINGS = {
    "C01": {
        ("data/clean/sample00022.json", 119.759, 256.067),
        ("data/clean/sample00031.json", 85.216, 184.868),
    },
    "C02": {
        ("data/clean/sample00047.json", 11584.162, 11663.592),
        ("data/clean/sample00003.json", 4.604, 58.103),
    },
    "C03": {("data/clean/sample00052.json", 447.434, 501.512)},
    "F02": {("data/clean/sample00035.json", 422.037, 448.113)},
    "R01": {
        ("data/clean/sample00051.json", 142.852, 180.323),
        ("data/clean/sample00051.json", 273.592, 309.35),
        ("data/clean/sample00051.json", 348.97, 393.252),
        ("data/clean/sample00051.json", 417.502, 421.94),
    },
    "R02": {
        ("data/clean/sample00014.json", 856.893, 879.393),
        ("data/clean/sample00014.json", 1277.502, 1343.298),
    },
}


def test_v1_fixture_has_exact_case_set_and_no_rejected_cards():
    suite = load_suite(FIXTURE_PATH)

    actual = [
        (case.id, case.kind, case.question, case.card_ids) for case in suite.cases
    ]
    assert actual == EXPECTED_CASES
    used = {card_id for case in suite.cases for card_id in case.card_ids}
    assert not used & {"card::1676", "card::1138", "card::1632"}
    assert all(case.retrieval_relevant_ids == case.card_ids for case in suite.cases)


def test_v1_fixture_has_exact_stock_generation_policy_and_stable_hashes():
    suite = load_suite(FIXTURE_PATH)

    assert suite.schema_version == 1
    assert suite.suite_id == "stock-qwen3-card-eval-v1"
    assert asdict(suite.model_policy) == {
        "required_model": "qwen3:14b",
        "forbid_abliterated": True,
    }
    assert asdict(suite.diagnostic_generation) == {
        "think": False,
        "temperature": 0,
        "seed": 20260711,
        "top_p": 1.0,
        "top_k": 1,
        "repeat_penalty": 1.05,
        "num_ctx": 4096,
        "num_predict": 384,
        "stream": False,
        "keep_alive": 0,
    }
    assert asdict(suite.production_generation) == {
        "think": False,
        "temperature": 0.7,
        "seeds": (20260711, 20260712, 20260713),
        "top_p": 0.85,
        "repeat_penalty": 1.2,
        "num_ctx": 4096,
        "num_predict": 640,
        "stream": False,
        "keep_alive": 0,
    }

    assert tuple((case.id, case.sha256) for case in suite.cases) == (
        EXPECTED_CASE_SHA256
    )
    assert suite.sha256 == EXPECTED_SUITE_SHA256


def test_v1_fixture_has_corrected_bindings_and_complete_provenance():
    suite = load_suite(FIXTURE_PATH)
    cases = {case.id: case for case in suite.cases}

    for case_id, expected in CORRECTED_BINDINGS.items():
        actual = {
            (evidence.clean_file, evidence.start, evidence.end)
            for evidence in cases[case_id].evidence
        }
        assert actual == expected

    for case in suite.cases:
        evidence_ids = {evidence.ref_id for evidence in case.evidence}
        assert all(atom.provenance_ref_ids for atom in case.atoms)
        assert all(
            set(atom.provenance_ref_ids) <= evidence_ids for atom in case.atoms
        )

    report = validate_suite(suite, REPO_ROOT)
    assert report.ok, "\n".join(report.errors)
