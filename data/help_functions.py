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
    phrase = clean_text(phrase)
    doc = nlp(phrase)
    lemmatized = ' '.join(
        token.lemma_ for token in doc
        if not token.is_punct
        and not token.is_space
        and token.pos_ not in ["ADP", "DET", "CCONJ", "SCONJ", "PART", "PRON"]
    )
    return remove_diacritics(lemmatized).lower()


def compute_scor_medical_diabet(row):
    scor = 0

    # === 1. Factori demografici și antropometrici ===
    if row.get("Vârstă", 0) > 45:
        scor += 1
    if row.get("IMC", 0) > 30:
        scor += 2
    if row.get("obezitate abdominala", 0) == 1:
        scor += 2

    # === 2. Diagnostic sau condiții medicale ===
    scor += int(row.get("rezistenta la insulina", 0)) * 4
    scor += int(row.get("prediabet", 0)) * 6
    scor += int(row.get("diabet zaharat tip 2", 0)) * 10
    scor += int(row.get("ficat gras", 0)) * 2

    # === 3. Simptome subiective și comportamente metabolice ===
    scor += int(row.get("slăbesc greu", 0))
    scor += int(row.get("mă îngraș ușor", 0))
    scor += int(row.get("depun grasime in zona abdominala", 0)) * 2
    scor += int(row.get("urinare nocturna", 0)) * 2
    scor += int(row.get("lipsa de energie", 0)) * 2
    scor += int(row.get("pofte de dulce", 0)) * 2
    scor += int(row.get("foame greu de controlat", 0)) * 2

    # === 4. Sex-specific ===
    sex = str(row.get("Ești", "")).strip().lower()
    talie = row.get("Care este circumferința taliei tale, măsurata deasupra de ombilicului?", 0)

    try:
        talie_val = float(talie)
    except:
        talie_val = 0  # în caz că e NaN sau text

    if sex == "femeie":
        scor += int(row.get("sindromul ovarelor polichistice", 0)) * 2
        if isinstance(row.get("labels"), list) and "ginecologic_hormonal" in row["labels"]:
            scor += 2
        if talie_val > 80:
            scor += 2
        if talie_val > 100:
            scor += 3

    elif sex == "barbat":
        if talie_val > 94:
            scor += 2
        if talie_val > 110:
            scor += 3  # prag pentru bărbați

    # === 5. Etichete NLP (semnale indirecte de risc) ===
    if isinstance(row.get("labels"), list):
        scor += 4 if "metabolic_endocrin" in row["labels"] else 0
        scor += 1 if "gastro_hepato_renal" in row["labels"] else 0
        scor += 1 if "inflamator_autoimun" in row["labels"] else 0
        scor += 1 if "neuro_psiho_energie" in row["labels"] else 0

    return scor


threshold_prediabet = 25
threshold_diabet = 30


def infer_diagnosis_from_scor(row):
    if row.get("rezistenta la insulina", 0) == 0 and \
            row.get("prediabet", 0) == 0 and \
            row.get("diabet zaharat tip 2", 0) == 0:

        scor = row["scor_medical"]
        if scor >= threshold_diabet:
            row["diabet zaharat tip 2"] = 1
        elif scor >= threshold_prediabet:
            row["prediabet"] = 1
    return row


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
