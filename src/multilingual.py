"""
src/multilingual.py
-------------------
Comprehensive Hinglish and Gujlish Clinical Language Engine for Medical AI.

Capabilities:
1. Language Detection:
   - Identifies whether medical input is English, Hinglish (Hindi in Latin script),
     or Gujlish (Gujarati in Latin script).
2. Clinical Translation & Normalization:
   - Converts colloquial Hinglish & Gujlish symptoms, body parts, and disease descriptors
     into standardized medical English terminology so the TF-IDF vectorizer and ML models
     can accurately classify disease domains.
3. Multilingual Laboratory & Ultrasound Explanations:
   - Provides plain-language explanations of laboratory deviations, clinical acuity,
     and ultrasound findings in English, Hinglish, and Gujlish.
"""

import re
from typing import Dict, List, Any, Tuple, Optional

# -----------------------------------------------------------------------------
# 1. Language Detection Markers
# -----------------------------------------------------------------------------

GUJLISH_MARKERS = {
    # Unambiguous Gujarati grammatical particles & vocabulary in Latin script
    "che", "chhe", "nathi", "aavyu", "aavi", "aavya", "aavse", "hato", "hati", "hata",
    "karvu", "karva", "karyu", "thay", "thayu", "thaye", "thase", "thavanu",
    "mare", "tamare", "mane", "tamne", "maro", "mari", "maru", "mara", "tamaro", "tamari",
    "ane", "mate", "dardi", "dukhavo", "lakshan", "vadhare", "ochhu", "ochho", "ochhi",
    "saru", "joye", "khabar", "aave", "rahyu", "ghano", "ghani", "ghanu",
    "lohi", "kamlo", "pathri", "chhati", "dhabkara", "magaj", "mutra",
    "sardi", "khasi", "taav", "kasrat", "dava", "bhanvanu", "khorak",
    "chaada", "jhada", "pachatu", "lakwo", "aakha", "sharir",
    "tapaas", "bajari", "chanda", "jantuo", "kankari"
}

HINGLISH_MARKERS = {
    # Unambiguous Hindi grammatical particles & vocabulary in Latin script
    "hai", "hain", "nahi", "nahin", "hoga", "hogi", "hoge", "hua", "hui", "huye",
    "mera", "meri", "mere", "mujhe", "mujhko", "tera", "teri", "tere", "tujhe",
    "unka", "unki", "unke", "iska", "iski", "iske", "aaya", "aayi", "aaye",
    "raha", "rahi", "rahe", "karta", "karti", "karte", "karna", "kiya", "kiye",
    "aur", "lekin", "kyunki", "isliye", "bahut", "bohot", "zyada",
    "dard", "seene", "bukhar", "khoon", "peshab", "ulti", "jalan", "kamzori",
    "thakan", "thaka", "dil", "dimag", "ilaaj", "dawai",
    "pait", "soojan", "chakkar", "saans", "khansi",
    "peeliya", "piliya", "pitte", "gurde", "lakwa", "dhadkan", "chalu"
}

def detect_language(text: str) -> Dict[str, Any]:
    """
    Detects whether input text is English, Hinglish, or Gujlish.
    Evaluates lexical density of Gujarati and Hindi transliteration markers.
    """
    if not text or not isinstance(text, str):
        return {
            "language_code": "en",
            "language_name": "English",
            "confidence": 1.0,
            "gujlish_score": 0,
            "hinglish_score": 0
        }

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return {
            "language_code": "en",
            "language_name": "English",
            "confidence": 1.0,
            "gujlish_score": 0,
            "hinglish_score": 0
        }

    total_words = len(words)
    guj_matches = [w for w in words if w in GUJLISH_MARKERS]
    hin_matches = [w for w in words if w in HINGLISH_MARKERS]

    guj_count = len(guj_matches)
    hin_count = len(hin_matches)

    # Threshold checks
    if guj_count > hin_count and guj_count >= 1:
        confidence = min(round((guj_count / max(total_words, 4)) * 2.5, 2), 0.99)
        return {
            "language_code": "gujlish",
            "language_name": "Gujlish (Gujarati in English letters)",
            "confidence": max(confidence, 0.70),
            "matched_markers": list(set(guj_matches))[:5],
            "gujlish_score": guj_count,
            "hinglish_score": hin_count
        }
    elif hin_count >= 1:
        confidence = min(round((hin_count / max(total_words, 4)) * 2.5, 2), 0.99)
        return {
            "language_code": "hinglish",
            "language_name": "Hinglish (Hindi in English letters)",
            "confidence": max(confidence, 0.70),
            "matched_markers": list(set(hin_matches))[:5],
            "gujlish_score": guj_count,
            "hinglish_score": hin_count
        }
    else:
        return {
            "language_code": "en",
            "language_name": "English",
            "confidence": 0.95,
            "matched_markers": [],
            "gujlish_score": 0,
            "hinglish_score": 0
        }


# -----------------------------------------------------------------------------
# 2. Hinglish & Gujlish to Medical English Translation & Normalization Map
# -----------------------------------------------------------------------------

# Mapping of colloquial multi-word phrases and single words into target medical keywords
CLINICAL_TRANSLATION_MAP = [
    # --- Cardiovascular (Category 4) ---
    (r"\b(?:chhati\s+ma\s+dukhavo|chhati\s+dukhe\s+che|chhati\s+ma\s+dard)\b", "acute chest pain angina myocardial infarction cardiovascular"),
    (r"\b(?:seene\s+me\s+dard|seene\s+me\s+jalan|chhati\s+me\s+dard)\b", "acute chest pain angina myocardial infarction cardiovascular"),
    (r"\b(?:heart\s+attack|dil\s+ka\s+daura|hriday\s+rog|dil\s+ki\s+bimari)\b", "acute myocardial infarction coronary artery cardiac failure"),
    (r"\b(?:dhabkara\s+vadhi\s+gaya|dhadkan\s+tez|dil\s+ki\s+dhadkan)\b", "cardiac arrhythmia tachycardia palpitations"),
    (r"\b(?:saans\s+phoolna|saans\s+lene\s+me\s+dikkat|shwas\s+levi\s+ma\s+taklif)\b", "dyspnea shortness of breath pulmonary congestion heart failure"),
    (r"\b(?:bp\s+vadhare|high\s+bp|lohi\s+nu\s+daban|rakt\s+chap)\b", "systemic arterial hypertension vascular pressure"),
    (r"\b(?:nas\s+block|artery\s+block|khoon\s+ki\s+nadi\s+block)\b", "coronary artery stenosis vascular occlusion thrombosis"),
    (r"\b(?:hriday|chhati|seena|dhadkan)\b", "cardiac cardiovascular coronary"),

    # --- Neoplasms / Oncology (Category 1) ---
    (r"\b(?:cancer|cancor|kencar)\b", "malignant neoplasm carcinoma tumor oncology"),
    (r"\b(?:tumor|gaanth|gaath|rasoli|masodu)\b", "neoplasm tumor carcinoma malignant mass biopsy"),
    (r"\b(?:chemo|chemotherapy|radiation)\b", "chemotherapy oncology cytotoxic antineoplastic"),
    (r"\b(?:biopsy\s+karai|biopsy\s+karavi)\b", "histopathology biopsy malignancy tissue diagnostic"),
    (r"\b(?:blood\s+cancer|lohi\s+nu\s+cancer|khoon\s+ka\s+cancer)\b", "leukemia hematologic malignancy neoplasm"),
    (r"\b(?:breast\s+cancer|stan\s+nu\s+cancer|stan\s+me\s+gaanth)\b", "breast carcinoma mammary neoplasm malignancy"),
    (r"\b(?:koshio\s+vadhe\s+che|cells\s+badh\s+rahe\s+hai)\b", "cellular hyperproliferation malignancy neoplastic growth"),

    # --- Digestive / Gastroenterology (Category 2) ---
    (r"\b(?:pet\s+ma\s+dukhavo|pet\s+dukhe\s+che|pait\s+me\s+dard|pet\s+me\s+dard)\b", "acute abdominal pain gastric gastrointestinal digestive"),
    (r"\b(?:pet\s+ma\s+pathri|gallbladder\s+ma\s+pathri|pitte\s+ki\s+thaili\s+me\s+pathri)\b", "cholelithiasis gallbladder calculi biliary colic digestive"),
    (r"\b(?:liver\s+ma\s+charbi|liver\s+par\s+charbi|fatty\s+liver)\b", "diffuse hepatic steatosis nonalcoholic fatty liver digestive"),
    (r"\b(?:kamlo|piliya|peeliya)\b", "jaundice hyperbilirubinemia hepatic biliary hepatitis"),
    (r"\b(?:zada|jhada|dast|loose\s+motion|pet\s+kharab)\b", "diarrhea gastroenteritis gastrointestinal digestive"),
    (r"\b(?:ulti|vomiting|oqyan)\b", "emesis vomiting gastric dyspepsia"),
    (r"\b(?:pet\s+me\s+ulcer|chanda|antarda\s+ma\s+chanda)\b", "peptic gastric ulcer duodenal gastrointestinal"),
    (r"\b(?:acidity|pait\s+me\s+jalan|chhati\s+me\s+jalan)\b", "gastroesophageal reflux dyspepsia gastritis acid"),
    (r"\b(?:khana\s+hazam\s+nahi|khorak\s+pachatu\s+nathi)\b", "malabsorption dyspepsia indigestion gastrointestinal"),

    # --- Nervous System / Neurology (Category 3) ---
    (r"\b(?:lakwo\s+mari\s+gayo|lakwa\s+mar\s+gaya|paralysis)\b", "cerebrovascular stroke hemiplegia paralysis neurological"),
    (r"\b(?:magaj\s+ni\s+nas\s+fati|dimag\s+ki\s+nas\s+phat|brain\s+stroke)\b", "cerebral hemorrhage ischemic stroke neurology"),
    (r"\b(?:mirgi|khech\s+aavi|daure\s+padna|daura)\b", "epileptic seizure convulsions neurological dysfunction"),
    (r"\b(?:mathu\s+dukhe\s+che|sar\s+me\s+dard|sir\s+dard|migraine)\b", "cephalalgia acute migraine chronic headache neurological"),
    (r"\b(?:chakkar\s+aave\s+che|chakkar\s+aana|chakkar)\b", "vertigo dizziness vestibular cerebellar neurological"),
    (r"\b(?:haath\s+pag\s+sunn|hath\s+paon\s+sunn|sunn\s+padvu)\b", "peripheral neuropathy paresthesia sensory nerve deficit"),
    (r"\b(?:bholvano\s+rog|yaaddasht\s+kamzor|bhoolna)\b", "dementia memory loss cognitive neurological impairment"),
    (r"\b(?:magaj|dimag|mastishk)\b", "cerebral neurological cranial brain"),

    # --- General Pathology / Systemic (Category 5) ---
    (r"\b(?:taav\s+aave\s+che|taav|bukhar|bukhaar)\b", "pyrexia febrile fever systemic inflammatory pathology"),
    (r"\b(?:thandi\s+vaye\s+che|kampan|shivering)\b", "chills rigors systemic pyogenic infection"),
    (r"\b(?:aakha\s+sharir\s+ma\s+infection|poore\s+sharir\s+me\s+infection|sepsis)\b", "systemic sepsis bacteremia generalized pathological condition"),
    (r"\b(?:sujan\s+che|sujan|soojan|sooj\s+gaya)\b", "edema inflammatory swelling tissue pathology"),
    (r"\b(?:ghav\s+rooz\s+nathi|ghav\s+nahi\s+bhar\s+raha|zakham)\b", "impaired wound healing tissue ulceration necrosis pathology"),
    (r"\b(?:lohi\s+ni\s+kami|khoon\s+ki\s+kami|anemia)\b", "severe anemia hemoglobin deficit hematologic pathology"),
    (r"\b(?:bahut\s+thakan|ashakti|kamzori|ashakt)\b", "chronic fatigue systemic debility constitutional condition"),
    (r"\b(?:jantuo\s+nu\s+sankraman|keeda|keede)\b", "pathogenic bacterial parasitic infection"),

    # General Words
    (r"\b(?:lohi|khoon)\b", "blood hemoglobin hematology"),
    (r"\b(?:mutra|peshab)\b", "urinary renal kidney"),
    (r"\b(?:dardi|mariz|patient)\b", "patient clinical subject"),
    (r"\b(?:dava|dawai|goli|medicine)\b", "pharmacotherapy medication therapy"),
    (r"\b(?:tapaas|jaanch|report|test)\b", "investigation diagnostic laboratory evaluation")
]

def translate_clinical_text(text: str) -> Dict[str, Any]:
    """
    Translates Hinglish or Gujlish clinical text into standard English medical terminology.
    Returns:
    - original_text: Input text
    - detected_language: Language metadata
    - translated_text: Text normalized to medical English keywords
    - substitutions_made: Count of matched clinical concepts
    """
    lang_info = detect_language(text)
    if lang_info["language_code"] == "en":
        return {
            "original_text": text,
            "detected_language": lang_info,
            "translated_text": text,
            "substitutions_made": 0,
            "is_translated": False
        }

    translated = text
    subs_count = 0

    for pattern, replacement in CLINICAL_TRANSLATION_MAP:
        matches = re.findall(pattern, translated, re.IGNORECASE)
        if matches:
            subs_count += len(matches)
            translated = re.sub(pattern, " " + replacement + " ", translated, flags=re.IGNORECASE)

    # Normalize whitespace
    translated = re.sub(r"\s+", " ", translated).strip()

    return {
        "original_text": text,
        "detected_language": lang_info,
        "translated_text": translated,
        "substitutions_made": subs_count,
        "is_translated": True
    }


# -----------------------------------------------------------------------------
# 3. Multilingual Explanations for Laboratory Parameters
# -----------------------------------------------------------------------------

MULTILINGUAL_LAB_EXPLANATIONS = {
    "hemoglobin": {
        "LOW": {
            "en": "Low hemoglobin indicates Anemia (decreased oxygen-carrying capacity).",
            "hinglish": "Hemoglobin kam hai (Anemia) - sharir me khoon aur oxygen pahunchane ki kshamta kam ho gayi hai. Hari sabziyan aur iron lein.",
            "gujlish": "Hemoglobin ochhu che (Anemia) - sharir ma lohi ane oxygen ni kami darshave che. Leeli bhaaji ane iron yukt khorak levano che."
        },
        "HIGH": {
            "en": "Elevated hemoglobin suggests Polycythemia, chronic hypoxia, or dehydration.",
            "hinglish": "Hemoglobin zyada hai - Polycythemia, dehydration ya oxygen ki kami se khoon gaadha ho sakta hai.",
            "gujlish": "Hemoglobin vadhare che - dehydration athva lohi ghaadhu thavanu suchve che. Paani vadhu pivo."
        },
        "NORMAL": {
            "en": "Optimal oxygen-carrying red cell pigment.",
            "hinglish": "Hemoglobin normal hai - sharir me khoon ki matra bilkul theek hai.",
            "gujlish": "Hemoglobin normal che - sharir ma lohi nu praman ekdam barabar che."
        }
    },
    "wbc": {
        "LOW": {
            "en": "Leukopenia: immune system suppression, viral infection, or bone marrow underactivity.",
            "hinglish": "WBC count kam hai - immunity kamzor hai ya koi viral infection hai.",
            "gujlish": "WBC count ochho che - rog pratikarak shakti ochhi che athva viral taav che."
        },
        "HIGH": {
            "en": "Leukocytosis: active bacterial infection, acute systemic inflammation, or physiological stress.",
            "hinglish": "WBC count badha hua hai - sharir me active bacterial infection ya sujan (inflammation) hai.",
            "gujlish": "WBC count vadhare che - sharir ma active infection athva sujan darshave che."
        },
        "NORMAL": {
            "en": "Immune white blood cell defense is balanced.",
            "hinglish": "WBC count normal hai - infection se ladne wale cells santulit hain.",
            "gujlish": "WBC count normal che - rog pratikarak koshio barabar che."
        }
    },
    "platelets": {
        "LOW": {
            "en": "Thrombocytopenia: increased bleeding and bruising vulnerability (e.g. viral fevers / dengue).",
            "hinglish": "Platelets kam hain (Thrombocytopenia) - khoon jamne ki taqat kam hai, dengue ya viral fever me hota hai.",
            "gujlish": "Platelets ochha che (Thrombocytopenia) - lohi jami javani shakti ghate che, dengue ma aam thay che."
        },
        "HIGH": {
            "en": "Thrombocytosis: reactive inflammation or acute blood loss recovery.",
            "hinglish": "Platelets zyada hain - reactive sujan ya iron deficiency ka sanket ho sakta hai.",
            "gujlish": "Platelets vadhare che - sharir ma sujan athva iron ni kami suchve che."
        },
        "NORMAL": {
            "en": "Adequate platelet clotting capability.",
            "hinglish": "Platelet count normal hai - khoon ka thakka theek se jam sakta hai.",
            "gujlish": "Platelets normal che - lohi jami javani prakriya barabar che."
        }
    },
    "glucose_fasting": {
        "LOW": {
            "en": "Hypoglycemia: low blood sugar requiring carbohydrate correction.",
            "hinglish": "Fasting blood sugar kam hai (Hypoglycemia) - kamzori aur chakkar aa sakte hain, glucose lein.",
            "gujlish": "Fasting sugar ochhu che (Hypoglycemia) - ashakti ane chakkar aavi shake che, thoduk galyu khorak lo."
        },
        "HIGH": {
            "en": "Hyperglycemia: 100-125 mg/dL indicates Prediabetes; >=126 mg/dL indicates Diabetes Mellitus.",
            "hinglish": "Fasting sugar zyada hai (Hyperglycemia) - Prediabetes ya Diabetes ka sanket hai, meetha kam karein.",
            "gujlish": "Fasting sugar vadhare che (Hyperglycemia) - Diabetes athva Prediabetes darshave che, galyu bandh karo."
        },
        "NORMAL": {
            "en": "Fasting glucose regulation is normal.",
            "hinglish": "Fasting blood sugar normal hai - sharir me sugar control bilkul theek hai.",
            "gujlish": "Fasting sugar normal che - sharir ma sugar control ekdam barabar che."
        }
    },
    "hba1c": {
        "LOW": {
            "en": "Low average blood sugar.",
            "hinglish": "HbA1c kam hai - pichle 3 mahine ka average sugar thoda kam hai.",
            "gujlish": "HbA1c ochhu che - pichla 3 mahina nu average sugar ochhu che."
        },
        "HIGH": {
            "en": "5.7 - 6.4% indicates Prediabetes; >=6.5% confirms chronic Diabetes Mellitus (3-month record).",
            "hinglish": "HbA1c high hai (>=6.5%) - pichle 3 mahine ka Diabetes record uncha hai. Doctor se davai lein.",
            "gujlish": "HbA1c vadhare che (>=6.5%) - 3 mahina no average sugar record Diabetes darshave che. Doctor ni salah lo."
        },
        "NORMAL": {
            "en": "Healthy 3-month glycemic equilibrium.",
            "hinglish": "HbA1c normal hai - pichle 3 mahine me sugar balance bahut achha raha.",
            "gujlish": "HbA1c normal che - pichla 3 mahina ma sugar control khub saras rahyu."
        }
    },
    "creatinine": {
        "LOW": {
            "en": "Low muscle mass or malnutrition.",
            "hinglish": "Creatinine kam hai - muscle mass kam ya protein ki kami ho sakti hai.",
            "gujlish": "Creatinine ochhu che - muscle mass ochhu athva protein ni kami suchve che."
        },
        "HIGH": {
            "en": "Elevated Creatinine indicates impaired renal glomerular filtration or kidney dysfunction.",
            "hinglish": "Serum Creatinine high hai - Gurde (Kidney) me filter karne me takleef hai. Urologist/Nephrologist se milein.",
            "gujlish": "Creatinine vadhare che - Kidneys barabar filter nathi kari rahi. Doctor ni tatkal salah lo."
        },
        "NORMAL": {
            "en": "Kidney filtration function is intact.",
            "hinglish": "Creatinine normal hai - Kidneys theek se kaam kar rahi hain.",
            "gujlish": "Creatinine normal che - Kidneys ekdam barabar kaam kare che."
        }
    },
    "total_cholesterol": {
        "LOW": {
            "en": "Severe malnutrition or liver insufficiency.",
            "hinglish": "Cholesterol kam hai - poshan ki kami ya liver kamzori.",
            "gujlish": "Cholesterol ochhu che - poshan ni kami darshave che."
        },
        "HIGH": {
            "en": "Hypercholesterolemia: contributes to coronary plaque formation and cardiovascular risk.",
            "hinglish": "Total Cholesterol high hai - naso me charbi jamne aur dil ke daure ka risk badhata hai. Tel-masale kam karein.",
            "gujlish": "Cholesterol vadhare che - hriday ni naso ma charbi jami shake che. Tel ane talelu khorak ochho karo."
        },
        "NORMAL": {
            "en": "Optimal total cholesterol concentration.",
            "hinglish": "Cholesterol normal hai - dil ke liye safe limit me hai.",
            "gujlish": "Cholesterol normal che - hriday mate surakshit maryada ma che."
        }
    },
    "sgpt_alt": {
        "LOW": {
            "en": "Normal finding.",
            "hinglish": "ALT normal hai.",
            "gujlish": "ALT normal che."
        },
        "HIGH": {
            "en": "Hepatocellular Injury / Transaminitis: active liver inflammation or steatosis (fatty liver).",
            "hinglish": "SGPT / ALT high hai - Liver me sujan (inflammation) ya fatty liver ka sanket hai.",
            "gujlish": "SGPT / ALT vadhare che - Liver ma sujan athva charbi (fatty liver) darshave che."
        },
        "NORMAL": {
            "en": "Liver cellular integrity is maintained.",
            "hinglish": "SGPT normal hai - Liver theek hai.",
            "gujlish": "SGPT normal che - Liver barabar kaam kare che."
        }
    },
    "bilirubin_total": {
        "LOW": {
            "en": "Normal finding.",
            "hinglish": "Bilirubin normal hai.",
            "gujlish": "Bilirubin normal che."
        },
        "HIGH": {
            "en": "Hyperbilirubinemia (Jaundice): caused by biliary obstruction, hepatitis, or hemolysis.",
            "hinglish": "Bilirubin high hai (Peeliya / Jaundice) - Aankhein aur peshab peela ho sakta hai. Liver checkup karayein.",
            "gujlish": "Bilirubin vadhare che (Kamlo / Jaundice) - Aankho ane peshab pilu thay shake. Liver ni tapanio karavo."
        },
        "NORMAL": {
            "en": "Normal biliary excretion.",
            "hinglish": "Bilirubin normal hai - Jaundice nahi hai.",
            "gujlish": "Bilirubin normal che - Kamlo nathi."
        }
    },
    "tsh": {
        "LOW": {
            "en": "Suppressed TSH indicates Hyperthyroidism (overactive thyroid).",
            "hinglish": "TSH kam hai (Hyperthyroidism) - Thyroid zyada active hai, dhadkan badhna aur vajan ghatna ho sakta hai.",
            "gujlish": "TSH ochhu che (Hyperthyroidism) - Thyroid vadhu active che, vajan ghati shake ane dhabkara vadhi shake."
        },
        "HIGH": {
            "en": "Elevated TSH indicates Primary Hypothyroidism (underactive thyroid).",
            "hinglish": "TSH high hai (Hypothyroidism) - Thyroid kamzor hai, thakan, vajan badhna aur thand lagna hota hai.",
            "gujlish": "TSH vadhare che (Hypothyroidism) - Thyroid ochhu kaam kare che, thakan ane vajan vadhi shake che."
        },
        "NORMAL": {
            "en": "Pituitary-thyroid feedback is normal.",
            "hinglish": "TSH normal hai - Thyroid theek kaam kar raha hai.",
            "gujlish": "TSH normal che - Thyroid ekdam barabar che."
        }
    }
}

def get_multilingual_lab_explanation(test_key: str, status: str, lang: str = "en") -> str:
    """Returns test explanation in English, Hinglish, or Gujlish."""
    lang_key = lang.lower() if lang in ["en", "hinglish", "gujlish"] else "en"
    test_data = MULTILINGUAL_LAB_EXPLANATIONS.get(test_key, {})
    status_data = test_data.get(status.upper(), {})
    return status_data.get(lang_key, status_data.get("en", "Value evaluated against reference range."))


# -----------------------------------------------------------------------------
# 4. Multilingual Explanations for Sonography / Ultrasound Findings
# -----------------------------------------------------------------------------

MULTILINGUAL_SONO_EXPLANATIONS = {
    "Cholelithiasis (Gallbladder Calculi)": {
        "en": "Mobile calcified stones present in the gallbladder lumen with acoustic shadowing.",
        "hinglish": "Pitte ki thaili (Gallbladder) me pathri (stones) maujood hai. Agar pet me dard ho toh surgeon se milein.",
        "gujlish": "Gallbladder ma pathri (stones) aavi che. Jo pet ma dukhavo thay toh doctor no sampark karvo."
    },
    "Diffuse Hepatic Steatosis (Fatty Liver)": {
        "en": "Increased parenchymal echogenicity indicating fat accumulation in liver hepatocytes.",
        "hinglish": "Liver me charbi jama ho gayi hai (Fatty Liver). Vajan kam karein aur oily khana bandh karein.",
        "gujlish": "Liver ma charbi jama thai gayi che (Fatty Liver). Kasrat karo ane tel-valoo khorak ochho karo."
    },
    "Nephrolithiasis (Renal Calculus)": {
        "en": "Dense echogenic calculus observed in renal collecting system.",
        "hinglish": "Gurde (Kidney) me pathri dikhi hai. Khoob paani piyein aur urologist se check karayein.",
        "gujlish": "Kidney ma pathri aavi che. Vadhu paani pivo ane doctor ni salah lo."
    },
    "Hydronephrosis (Pelvicalyceal Dilatation)": {
        "en": "Urine outflow dilatation suggesting downstream ureteric obstruction.",
        "hinglish": "Kidney me sujan hai (Hydronephrosis) kyunki peshab ka rasta pathri se ruk raha hai.",
        "gujlish": "Kidney ma sujan aavi che (Hydronephrosis) peshab no marg aavrodh thayo hovathi."
    },
    "Renal Cortical Cyst": {
        "en": "Anechoic fluid-filled simple cortical cyst; benign finding.",
        "hinglish": "Kidney me paani ki normal gaanth (Simple Cyst) hai - yeh benign (safe) hoti hai.",
        "gujlish": "Kidney ma paani ni saadi gaath (Simple Cyst) che - aama ghabravani zaroor nathi."
    },
    "Free Fluid / Ascites Observed": {
        "en": "Pathological fluid pooling in peritoneal spaces requiring urgent clinical assessment.",
        "hinglish": "Pet me paani (Ascites/Free Fluid) bhar gaya hai. Turant doctor se check karayein.",
        "gujlish": "Pet ma paani bharaai gayu che (Ascites). Tatkal doctor no sampark karvo."
    }
}

def get_multilingual_sono_explanation(finding_title: str, lang: str = "en") -> str:
    """Returns sonography finding explanation in English, Hinglish, or Gujlish."""
    lang_key = lang.lower() if lang in ["en", "hinglish", "gujlish"] else "en"
    item = MULTILINGUAL_SONO_EXPLANATIONS.get(finding_title, {})
    return item.get(lang_key, item.get("en", "Radiological anatomical finding."))


# -----------------------------------------------------------------------------
# 5. Multilingual Acuity & Clinical Synthesis Labels
# -----------------------------------------------------------------------------

MULTILINGUAL_ACUITY = {
    "CRITICAL": {
        "en": "CRITICAL RISK",
        "hinglish": "GAMBHIR KHATRA (CRITICAL)",
        "gujlish": "GAMBHIR JO-KHAM (CRITICAL)"
    },
    "HIGH": {
        "en": "HIGH ATTENTION",
        "hinglish": "UCHH DRISHTI (HIGH ATTENTION)",
        "gujlish": "VADHU DHAYAN (HIGH ATTENTION)"
    },
    "MODERATE": {
        "en": "MODERATE MONITORING",
        "hinglish": "MADHYAM DEKHBHAL (MODERATE)",
        "gujlish": "MADHYAM TAPAAS (MODERATE)"
    },
    "NORMAL": {
        "en": "PHYSIOLOGICALLY NORMAL",
        "hinglish": "SAMAANYA / SAFE (NORMAL)",
        "gujlish": "SAMAANYA / SURAKSHIT (NORMAL)"
    }
}
