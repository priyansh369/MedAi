import pytest
from src.multilingual import (
    detect_language,
    translate_clinical_text,
    get_multilingual_lab_explanation,
    MULTILINGUAL_ACUITY
)


def test_detect_english():
    text = "The patient presents with severe chest pain and elevated cardiac troponin levels."
    lang = detect_language(text)
    assert lang["language_code"] == "en"


def test_detect_hinglish():
    text = "Khoon ki kami hai aur sugar bohot badha hua hai kamar me dard."
    lang = detect_language(text)
    assert lang["language_code"] == "hinglish"
    assert "Hinglish" in lang["language_name"]


def test_detect_gujlish():
    text = "Lohi oochu che ane pitte ma pathri che ghanu dard thay che."
    lang = detect_language(text)
    assert lang["language_code"] == "gujlish"
    assert "Gujlish" in lang["language_name"]


def test_detect_native_devanagari_hindi():
    text = "मरीज को बहुत तेज बुखार है और खून की कमी है।"
    lang = detect_language(text)
    assert lang["language_code"] == "hi"
    assert "Devanagari" in lang["language_name"]


def test_detect_native_gujarati_script():
    text = "દર્દીને લોહીની ઉણપ છે અને પિત્તાશયમાં પથરી જોવા મળે છે."
    lang = detect_language(text)
    assert lang["language_code"] == "gu"
    assert "Gujarati" in lang["language_name"]


def test_multilingual_translation_hinglish():
    text = "Khoon ki kami aur pet me dard bukhar hai"
    res = translate_clinical_text(text)
    assert res["substitutions_made"] > 0
    translated = res["translated_text"].lower()
    assert "anemia" in translated or "fever" in translated


def test_multilingual_translation_native_hindi():
    text = "रक्त शर्करा और हीमोग्लोबिन स्तर"
    res = translate_clinical_text(text)
    assert res["detected_language"]["language_code"] == "hi"


def test_multilingual_lab_explanations():
    expl_en = get_multilingual_lab_explanation("glucose_fasting", "HIGH", "en")
    expl_hi = get_multilingual_lab_explanation("glucose_fasting", "HIGH", "hinglish")
    expl_gu = get_multilingual_lab_explanation("glucose_fasting", "HIGH", "gujlish")

    assert "Hyperglycemia" in expl_en or "Diabetes" in expl_en
    assert "sugar" in expl_hi.lower() and ("zyada" in expl_hi.lower() or "diabetes" in expl_hi.lower())
    assert "sugar" in expl_gu.lower() and ("vadhare" in expl_gu.lower() or "diabetes" in expl_gu.lower())


def test_multilingual_acuity_mappings():
    assert "NORMAL" in MULTILINGUAL_ACUITY
    assert "CRITICAL" in MULTILINGUAL_ACUITY
    assert MULTILINGUAL_ACUITY["CRITICAL"]["hinglish"] != ""
    assert MULTILINGUAL_ACUITY["CRITICAL"]["gujlish"] != ""
