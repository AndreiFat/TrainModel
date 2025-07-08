# === 2. Funcții utilitare ===
import re
import unicodedata

import pandas as pd


def remove_diacritics(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.replace("ş", "ș").replace("ţ", "ț").lower()
    text = re.sub(r"[^a-zăîâșț0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return remove_diacritics(text)


def normalize_phrase(phrase, nlp):
    # doc = nlp(phrase)
    # lemmatized = ' '.join(
    #     token.lemma_ for token in doc
    #     if not token.is_punct
    #     and not token.is_space
    #     and token.pos_ not in ["ADP", "DET", "CCONJ", "SCONJ", "PART", "PRON"]
    # )
    phrase = clean_text(phrase)
    doc = nlp(phrase)

    tokens = [
        token.text for token in doc
        if not token.is_punct
           and not token.is_space
           and token.pos_ not in ["ADP", "DET", "CCONJ", "SCONJ", "PART", "PRON"]
    ]

    return remove_diacritics(' '.join(tokens)).lower()


def compute_scor_medical_diabet(row):
    scor = 0

    # === 1. Factori demografici și antropometrici ===
    varsta = row.get("Vârstă", 0)
    if varsta > 60:
        scor += 6
    elif varsta > 45:
        scor += 3
    elif varsta > 35:
        scor += 2

    imc = row.get("IMC", 0)
    if imc > 30:
        scor += 3
    elif imc > 25:
        scor += 2

    if row.get("obezitate abdominala", 0) == 1:
        scor += 2

    if row.get("ficat gras", 0) == 1:
        scor += 2

    severitate = {
        "rezistenta la insulina": 4,
        "prediabet": 6,
        "diabet zaharat tip 2": 8
    }
    max_scor = 0
    for afectiune, val in severitate.items():
        if int(row.get(afectiune, 0)) == 1:
            max_scor = max(max_scor, val)
    scor += max_scor

    # === 2. Simptome metabolice ===
    simptome = {
        "pofte de dulce": 2,
        "foame greu de controlat": 2,
        "depun grasime in zona abdominala": 2,
        "urinare nocturna": 1,
        "lipsa de energie": 1,
        "slăbesc greu": 1,
        "mă îngraș ușor": 1,
    }
    for key, pts in simptome.items():
        if int(row.get(key, 0)) == 1:
            scor += pts

    # === 3. Sex și circumferință talie ===
    sex = row.get("Ești ", -1)  # 0 = femeie, 1 = bărbat
    talie = row.get("Care este circumferința taliei tale, măsurata deasupra de ombilicului?", 0)

    if sex == 0:  # femeie
        if talie > 100:
            scor += 3
        elif talie > 80:
            scor += 2

        if int(row.get("sindromul ovarelor polichistice", 0)) == 1:
            scor += 2

        if isinstance(row.get("labels"), list) and "ginecologic_hormonal" in row["labels"]:
            scor += 2

    elif sex == 1:  # bărbat
        if talie > 110:
            scor += 3
        elif talie > 94:
            scor += 2

    # === 4. Etichete NLP adiționale (opțional) ===
    if isinstance(row.get("labels"), list):
        if "metabolic_endocrin" in row["labels"]:
            scor += 4
        if "gastro_hepato_renal" in row["labels"]:
            scor += 1
        if "inflamator_autoimun" in row["labels"]:
            scor += 1
        if "neuro_psiho_energie" in row["labels"]:
            scor += 2

    return scor


# threshold_prediabet = 15
# threshold_diabet = 20
#
#
# def infer_diagnosis_from_scor(row):
#     if row.get("rezistenta la insulina", 0) == 0 and \
#             row.get("prediabet", 0) == 0 and \
#             row.get("diabet zaharat tip 2", 0) == 0:
#
#         scor = row["scor_medical"]
#         if scor >= threshold_diabet:
#             row["diabet zaharat tip 2"] = 1
#         elif scor >= threshold_prediabet:
#             row["prediabet"] = 1
#     return row


def compute_scor_medical_cardio(row):
    scor = 0

    # === 1. Diagnostice cardiovasculare grave ===
    scor += int(row.get("infarct", 0)) * 12
    scor += int(row.get("avc", 0)) * 12
    scor += int(row.get("stent_sau_bypass", 0)) * 10

    # === 2. Alte afecțiuni cu impact cardiovascular ===
    scor += int(row.get("fibrilatie_sau_ritm", 0)) * 6
    scor += int(row.get("embolie_sau_tromboza", 0)) * 6

    # === 3. Condiții medicale preexistente ===
    scor += int(row.get("hipertensiune arteriala", 0)) * 6
    scor += int(row.get("dislipidemie (grăsimi crescute in sânge)", 0)) * 4

    # === 4. Factori diabetici ===
    scor += int(row.get("rezistenta la insulina", 0)) * 5
    scor += int(row.get("prediabet", 0)) * 7
    scor += int(row.get("diabet zaharat tip 2", 0)) * 10

    # === 5. Factori de risc subiectivi extrasi ===
    scor += int(row.get("risc_cardiovascular", 0)) * 2  # coloana calculata pe simptome/stil viata

    # === 6. Factori stil de viață și simptome indirecte ===
    scor += int(row.get("oboseala permanenta", 0)) * 2
    scor += int(row.get("lipsa de energie", 0)) * 2

    # === 7. Date antropometrice ===
    imc = row.get("IMC", 0)
    if imc > 30:
        scor += 3

    if row.get("Vârstă", 0) > 45:
        scor += 3

    if row.get("obezitate abdominala", 0) == 1:
        scor += 2

    sex = str(row.get("Ești", "")).strip().lower()
    try:
        talie = float(row.get("Care este circumferința taliei tale, măsurata deasupra de ombilicului?", 0))
    except:
        talie = 0

    if sex == "femeie" and talie > 80:
        scor += 2
    elif sex == "barbat" and talie > 94:
        scor += 2

    # === 8. NLP: etichete din text lemmatizat ===
    if isinstance(row.get("labels"), list):
        scor += 3 if "cardio_vascular" in row["labels"] else 0
        scor += 1 if "neuro_psiho_energie" in row["labels"] else 0

    return scor
