import ast
import json

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow.keras.backend as K
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score, \
    precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from scripts.mlp_model_diabetes.create_mlp_model import create_diabetes_model
from scripts.utils.load_data.load_data_utils import load_data


def plot_model_performance(history):
    metrics = ['loss', 'accuracy']
    plt.figure(figsize=(12, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 2, i)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f'{metric.upper()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.mean(K.sum(weight * cross_entropy, axis=1))

    return loss


def train_mlp_diabetes_model():
    # === 1. Încarcă datele
    train = load_data("data/datasets/train/train.csv", sep=';')
    val = load_data("data/datasets/validation/val.csv", sep=';')
    test = load_data("data/datasets/test/test.csv", sep=';')

    # === 2. Coloane de input
    feature_cols = [
        "Vârstă", "Ești ",
        "Care este greutatea ta actuala?", "Care este înălțimea ta? ",
        "Care este circumferința taliei tale, măsurata deasupra de ombilicului?",
        "IMC",
        "obezitate abdominala", "slăbesc greu", "mă îngraș ușor", "depun grasime in zona abdominala",
        "urinare nocturna", "pofte de dulce", "foame greu de controlat", "lipsa de energie",
        "ficat gras", "sindromul ovarelor polichistice",
        "scor_medical"
    ]

    continuous_cols = [
        "Vârstă", "Care este greutatea ta actuala?", "Care este înălțimea ta? ",
        "Care este circumferința taliei tale, măsurata deasupra de ombilicului?", "IMC",
        "scor_medical"
    ]
    binary_cols = list(set(feature_cols) - set(continuous_cols))

    # === 3. NLP labels → MultiLabel Binarizer
    for df_ in [train, val, test]:
        df_["labels"] = df_["labels"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        df_["labels"] = df_["labels"].apply(lambda x: ["none"] if len(x) == 0 else x)

    mlb_nlp = MultiLabelBinarizer()
    labels_train = pd.DataFrame(mlb_nlp.fit_transform(train["labels"]), columns=mlb_nlp.classes_)
    labels_val = pd.DataFrame(mlb_nlp.transform(val["labels"]), columns=mlb_nlp.classes_)
    labels_test = pd.DataFrame(mlb_nlp.transform(test["labels"]), columns=mlb_nlp.classes_)

    # === 4. Scalează doar coloanele continue
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train[continuous_cols]), columns=continuous_cols)
    val_scaled = pd.DataFrame(scaler.transform(val[continuous_cols]), columns=continuous_cols)
    test_scaled = pd.DataFrame(scaler.transform(test[continuous_cols]), columns=continuous_cols)

    # === 5. Concatenează features scalate + binare + NLP labels
    train_input = pd.concat([train_scaled, train[binary_cols].reset_index(drop=True), labels_train], axis=1)
    val_input = pd.concat([val_scaled, val[binary_cols].reset_index(drop=True), labels_val], axis=1)
    test_input = pd.concat([test_scaled, test[binary_cols].reset_index(drop=True), labels_test], axis=1)

    X_train = train_input.values
    X_val = val_input.values
    X_test = test_input.values
    print(np.bincount(train["diagnostic"].astype(int).values))
    print(np.bincount(val["diagnostic"].astype(int).values))
    print(np.bincount(test["diagnostic"].astype(int).values))
    print(train["diagnostic"].value_counts(normalize=True))
    print(val["diagnostic"].value_counts(normalize=True))
    print(test["diagnostic"].value_counts(normalize=True))
    print(X_test)
    print("Raw continuous input before scaling (test):", test[continuous_cols].iloc[0].values)

    # === 6. Pregătește y (diagnostic)
    y_train = to_categorical(train["diagnostic"].astype(int).values, num_classes=4)
    y_val = to_categorical(val["diagnostic"].astype(int).values, num_classes=4)
    y_test = to_categorical(test["diagnostic"].astype(int).values, num_classes=4)

    # === 7. Creează modelul
    input_dim = X_train.shape[1]
    model = create_diabetes_model(input_dim=input_dim, num_classes=4)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc')
        ]
    )

    # === 8. Calculare class weights
    y_train_raw = train["diagnostic"].astype(int).values
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train["diagnostic"].values),
        y=train["diagnostic"].values
    )
    class_weights = dict(enumerate(weights))

    # === 9. Antrenare model
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
    )

    # === 10. Plot performanță
    plot_model_performance(history)

    # === 11. Evaluare pe test
    y_pred_proba = model.predict(X_test)
    y_pred = y_pred_proba.argmax(axis=1)
    y_test_labels = y_test.argmax(axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test_labels, y_pred, digits=3))

    acc = accuracy_score(y_test_labels, y_pred)
    print(f"\n🔍 Acuratețea generală pe setul de test: {acc:.4f}")

    cm = confusion_matrix(y_test_labels, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["fără", "rezistență", "prediabet", "diabet"],
                yticklabels=["fără", "rezistență", "prediabet", "diabet"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Matrice de confuzie")
    plt.show()

    # === 12. Predicții individuale
    print("\n=== Predicții pentru primii 10 pacienți ===")
    label_map = {0: "fără", 1: "rezistență", 2: "prediabet", 3: "diabet"}
    for i in range(min(10, len(y_pred))):
        varsta = test.iloc[i]["Vârstă"]
        gen = "femeie" if test.iloc[i]["Ești "] == 0 else "bărbat"
        pred_label = label_map[y_pred[i]]
        prob = y_pred_proba[i][y_pred[i]]
        print(f"→ Pacient {i + 1} | Vârstă: {varsta} | Gen: {gen} | Predicție: {pred_label} ({prob * 100:.1f}%)")

    # === 13. Salvare model și artefacte
    model.save("models/mlp_model_multiclass/diagnostic_model.h5")

    # Presupunem că ai scalerul deja antrenat (fit)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
    }

    # Salvează în fișier JSON
    with open("models/mlp_model_multiclass/scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=4)

    dump(mlb_nlp, "models/mlp_model_multiclass/mlb_nlp.joblib")

    with open("models/mlp_model_multiclass/feature_cols.json", "w", encoding="utf-8") as f:
        json.dump(train_input.columns.tolist(), f, ensure_ascii=False, indent=4)

    print("Evaluare generală pe setul de test:")
    print(f"Acuratețe: {accuracy_score(y_test_labels, y_pred):.4f}")
    print(f"F1-score macro: {f1_score(y_test_labels, y_pred, average='macro'):.4f}")
    print(f"Precision macro: {precision_score(y_test_labels, y_pred, average='macro'):.4f}")
    print(f"Recall macro: {recall_score(y_test_labels, y_pred, average='macro'):.4f}")
    print(f"AUC macro: {roc_auc_score(y_test, y_pred_proba, average='macro', multi_class='ovo'):.4f}")

    # === Plot Acuratețe
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Acuratețe Antrenare')
    plt.plot(history.history['val_accuracy'], label='Acuratețe Validare')
    plt.xlabel('Epocă')
    plt.ylabel('Acuratețe')
    plt.title('Evoluția acurateții')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Pierdere Antrenare')
    plt.plot(history.history['val_loss'], label='Pierdere Validare')
    plt.xlabel('Epocă')
    plt.ylabel('Loss')
    plt.title('Evoluția funcției de pierdere')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
