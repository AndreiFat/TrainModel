import ast
import json

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam

from scripts.mlp_model_cardio.create_mlp_model import create_mlp_model
from scripts.utils.load_data.load_data_utils import load_data


def interpret_risk(prob):
    if prob < 0.25:
        return "Risc scăzut"
    elif prob < 0.6:
        return "Risc moderat"
    else:
        return "Risc ridicat"


def train_mlp_cardio():
    # === 1. Încarcă datele
    train = load_data("data/datasets/train/train.csv", sep=';')
    val = load_data("data/datasets/validation/val.csv", sep=';')
    test = load_data("data/datasets/test/test.csv", sep=';')

    # === 2. Coloane relevante
    feature_cols = [
        "Vârstă",
        "Ești ",
        "Care este greutatea ta actuala?", "Care este înălțimea ta? ",
        "Care este circumferința taliei tale, măsurata deasupra de ombilicului?",
        "IMC",
        "obezitate abdominala",
        # "slăbesc greu",
        # "mă îngraș ușor",
        # "depun grasime in zona abdominala",
        "oboseala permanenta",
        "lipsa de energie",
        # "urinare nocturna",
        # "pofte de dulce",
        # "foame greu de controlat",
        "ficat gras",
        "dislipidemie (grăsimi crescute in sânge)",
        "hipertensiune arteriala",
        "infarct",
        "avc",
        "stent_sau_bypass",
        "fibrilatie_sau_ritm",
        "embolie_sau_tromboza",
        # "risc_cardiovascular",
        "scor_medical_cardio",
    ]

    # === 3. Convertire sex
    for df_ in [train, val, test]:
        df_["Ești "] = df_["Ești "].map({"femeie": 1, "barbat": 0}).fillna(0).astype(int)

    # === 4. Transformă coloana `labels` folosind MultiLabelBinarizer
    mlb = MultiLabelBinarizer()

    # Convertim stringul în listă
    train_labels = train['labels'].apply(ast.literal_eval)
    val_labels = val['labels'].apply(ast.literal_eval)
    test_labels = test['labels'].apply(ast.literal_eval)

    # Fit pe train, transform pe toate
    mlb.fit(train_labels)

    train_mlb = pd.DataFrame(mlb.transform(train_labels), columns=mlb.classes_)
    val_mlb = pd.DataFrame(mlb.transform(val_labels), columns=mlb.classes_)
    test_mlb = pd.DataFrame(mlb.transform(test_labels), columns=mlb.classes_)

    # Concatenăm în dataframe-urile principale
    train = pd.concat([train.reset_index(drop=True), train_mlb], axis=1)
    val = pd.concat([val.reset_index(drop=True), val_mlb], axis=1)
    test = pd.concat([test.reset_index(drop=True), test_mlb], axis=1)

    # Adăugăm aceste coloane în feature_cols
    feature_cols += list(train_mlb.columns)

    # === 5. Pregătește X și y
    target_col = "risc_cardiovascular"

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_val = val[feature_cols]
    y_val = val[target_col]

    X_test = test[feature_cols]
    y_test = test[target_col]

    # === 6. Standardizare
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # === 6. Arată distribuția claselor
    def show_class_distribution(y, name):
        unique, counts = np.unique(y, return_counts=True)
        print(f"Distribuție clase în {name}: {dict(zip(unique, counts))}")

    show_class_distribution(y_train, "train")
    show_class_distribution(y_val, "validation")
    show_class_distribution(y_test, "test")

    # === 7. Antrenează modelul
    model = create_mlp_model(input_dim=X_train_scaled.shape[1])

    classes = np.array([0, 1])
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}

    optimizer = Adam(learning_rate=1e-4)

    # Callbacks: EarlyStopping + ReduceLROnPlateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           Precision(),
                           Recall()])

    model.fit(X_train_scaled, y_train,
              validation_data=(X_val_scaled, y_val),
              class_weight=class_weight_dict,
              epochs=10,
              batch_size=32,
              callbacks=[early_stopping, reduce_lr],
              )

    # === 8. Evaluează pe test
    val_loss, val_acc, val_prec, val_rec = model.evaluate(X_val_scaled, y_val)
    print(f"Val - Loss: {val_loss}, Acc: {val_acc}, Prec: {val_prec}, Rec: {val_rec}")

    # === 9. Predicții
    y_pred_proba = model.predict(X_test_scaled).flatten()
    y_pred_bin = (y_pred_proba > 0.5).astype(int)

    # === 10. Matrice de confuzie
    cm = confusion_matrix(y_test, y_pred_bin)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Reds", xticklabels=["Negativ", "Pozitiv"],
                yticklabels=["Negativ", "Pozitiv"])
    plt.title("Matrice de Confuzie - Test Set")
    plt.xlabel("Predicție")
    plt.ylabel("Adevăr")
    plt.tight_layout()
    plt.show()

    # === 11. Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_bin))

    # === 12. Primele 5 predicții interpretate
    print("\nPrimele 5 predicții și interpretări:")
    for i in range(50):
        prob = y_pred_proba[i]
        print(
            f"Persoana {i + 1}: risc = {prob * 100:.1f}% | clasă reală = {y_test.iloc[i]} | interpretare: {interpret_risk(prob)}")

    # model_rf = RandomForestClassifier()
    # model_rf.fit(X_train, y_train)
    #
    # importances = pd.Series(model_rf.feature_importances_, index=X_train.columns)
    # importances.sort_values().plot(kind='barh', figsize=(10, 8))
    # plt.title("Importanța variabilelor (Random Forest)")
    # plt.show()

    # === 13. Salvare model
    model.save("models/mlp_model_cardio/trained_mlp_cardio_risk.h5")
    joblib.dump(scaler, "models/mlp_model_cardio/scaler.joblib")
    joblib.dump(mlb, "models/mlp_model_cardio/mlb.joblib")

    with open("models/mlp_model_cardio/feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=4)
