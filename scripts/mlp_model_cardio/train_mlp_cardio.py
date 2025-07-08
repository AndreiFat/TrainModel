import ast
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from scripts.mlp_model_cardio.create_mlp_model import create_mlp_model


def interpret_risk(prob):
    if prob < 0.25:
        return "Risc scăzut"
    elif prob < 0.6:
        return "Risc moderat"
    else:
        return "Risc ridicat"


def train_mlp_cardio():
    # === 1. Încarcă datele
    train = pd.read_csv("data/datasets/train/train_cardio.csv", sep=";")
    val = pd.read_csv("data/datasets/validation/val_cardio.csv", sep=";")
    test = pd.read_csv("data/datasets/test/test_cardio.csv", sep=";")

    # === 2. Coloane folosite
    numeric_cols = [
        "Vârstă",
        "Care este greutatea ta actuala?",
        "Care este înălțimea ta? ",
        "Care este circumferința taliei tale, măsurata deasupra de ombilicului?",
        "IMC",
        "scor_medical_cardio"
    ]
    binary_cols = [
        "Ești ",  # gen: 0 = femeie, 1 = bărbat
        "obezitate abdominala",
        "rezistenta la insulina",
        "prediabet",
        "diabet zaharat tip 2",
        "oboseala permanenta",
        "lipsa de energie",
        "dislipidemie (grăsimi crescute in sânge)",
        "hipertensiune arteriala",
        "infarct",
        "avc",
        "stent_sau_bypass",
        "fibrilatie_sau_ritm",
        "embolie_sau_tromboza"
    ]
    target_col = "risc_cardiovascular"

    # === 3. Procesează etichetele NLP
    train_labels = train["labels"].apply(ast.literal_eval)
    val_labels = val["labels"].apply(ast.literal_eval)
    test_labels = test["labels"].apply(ast.literal_eval)

    mlb = MultiLabelBinarizer()
    mlb.fit(train_labels)

    train_nlp = pd.DataFrame(mlb.transform(train_labels), columns=mlb.classes_)
    val_nlp = pd.DataFrame(mlb.transform(val_labels), columns=mlb.classes_)
    test_nlp = pd.DataFrame(mlb.transform(test_labels), columns=mlb.classes_)

    # === 4. Concatenare NLP la datele originale
    train = pd.concat([train.reset_index(drop=True), train_nlp], axis=1)
    val = pd.concat([val.reset_index(drop=True), val_nlp], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_nlp], axis=1)

    nlp_cols = list(train_nlp.columns)  # toate etichetele NLP devin features

    feature_cols = numeric_cols + binary_cols + nlp_cols

    # === 5. Preprocesare
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(train[numeric_cols])
    X_val_num = scaler.transform(val[numeric_cols])
    X_test_num = scaler.transform(test[numeric_cols])

    X_train = np.concatenate([
        X_train_num,
        train[binary_cols + nlp_cols].values
    ], axis=1)

    X_val = np.concatenate([
        X_val_num,
        val[binary_cols + nlp_cols].values
    ], axis=1)

    X_test = np.concatenate([
        X_test_num,
        test[binary_cols + nlp_cols].values
    ], axis=1)

    y_train = train[target_col].astype(int).values
    y_val = val[target_col].astype(int).values
    y_test = test[target_col].astype(int).values

    # === 6. Model
    model = create_mlp_model(input_dim=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32
    )

    # === 7. Evaluare
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\nAcuratețe:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nPrimele 5 predicții și interpretări:")
    for i in range(5):
        prob = y_pred_proba[i]
        print(
            f"Persoana {i + 1}: risc = {prob * 100:.1f}% | clasă reală = {y_test[i]} | interpretare: {interpret_risk(prob)}"
        )

    # === 8. Matrice de confuzie
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap="Reds", fmt='d')
    plt.xlabel("Predicție")
    plt.ylabel("Adevăr")
    plt.title("Matrice de Confuzie")
    plt.show()

    # === 9. Acuratețe și loss
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Val acc')
    plt.legend()
    plt.title("Acuratețe")
    plt.show()

    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.legend()
    plt.title("Loss")
    plt.show()

    # === Salvare pentru FastAPI
    model.save("models/mlp_model_cardio/cardio_model.h5")
    joblib.dump(scaler, "models/mlp_model_cardio/cardio_scaler.pkl")
    joblib.dump(mlb, "models/mlp_model_cardio/cardio_mlb.pkl")
    with open("models/mlp_model_cardio/cardio_feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=4)

    # === 10. Importanță medie absolută
    import shap

    explainer = shap.Explainer(model, X_train, feature_names=feature_cols)
    shap_values = explainer(X_test)

    # Calculăm media absolută SHAP per feature
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": mean_abs_shap
    }).sort_values(by="importance", ascending=False)

    # === 11. Plot importanță
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=importance_df.head(20), palette="viridis")
    plt.title("Top 20 cele mai importante caracteristici (SHAP)")
    plt.xlabel("Importanță medie (|SHAP|)")
    plt.ylabel("Caracteristică")
    plt.tight_layout()
    plt.show()
