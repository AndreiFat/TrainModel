import ast
import json

import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from scripts.cnn_model_diabetes.create_cnn_model import create_diabetes_model
from scripts.utils.load_data.load_data_utils import load_data


def plot_model_performance(history):
    metrics = ['loss', 'precision', 'recall', 'auc']
    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        plt.title(f'{metric.upper()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()


def interpret_risk(prob):
    if prob < 0.3:
        return "risc scăzut"
    elif prob < 0.6:
        return "risc moderat"
    else:
        return "risc ridicat"


def train_mlp_diabetes_model():
    # === 1. Încarcă datele
    train = load_data("data/datasets/train/train.csv", sep=';')
    val = load_data("data/datasets/validation/val.csv", sep=';')
    test = load_data("data/datasets/test/test.csv", sep=';')

    feature_cols = [
        "Vârstă", "Ești ",  # ← adaugă sexul pentru logica SOP
        "Care este greutatea ta actuala?", "Care este înălțimea ta? ",
        "Care este circumferința taliei tale, măsurata deasupra de ombilicului?",
        "IMC",
        "obezitate abdominala", "slăbesc greu", "mă îngraș ușor", "depun grasime in zona abdominala",
        "oboseala permanenta", "urinare nocturna", "pofte de dulce",
        "foame greu de controlat", "lipsa de energie",
        "hipertensiune arteriala", "ficat gras", "dislipidemie (grăsimi crescute in sânge)",
        "sindromul ovarelor polichistice",
        "scor_medical",
        "labels"
    ]
    # Convertire sex (femeie = 1, bărbat = 0)
    for df_ in [train, val, test]:
        df_["Ești "] = df_["Ești "].map({"femeie": 1, "bărbat": 0}).fillna(0)

    # === 2. Etichete NLP → MultiLabel Binarizer
    train["labels"] = train["labels"].apply(ast.literal_eval)
    val["labels"] = val["labels"].apply(ast.literal_eval)
    test["labels"] = test["labels"].apply(ast.literal_eval)

    mlb_nlp = MultiLabelBinarizer()
    labels_train = pd.DataFrame(mlb_nlp.fit_transform(tr), columns=mlb_nlp.classes_)
    labels_val = pd.DataFrame(mlb_nlp.transform(val["labels"]), columns=mlb_nlp.classes_)
    labels_test = pd.DataFrame(mlb_nlp.transform(test["labels"]), columns=mlb_nlp.classes_)

    # === 3. Concatenează etichetele NLP
    train_input = pd.concat([train[feature_cols[:-1]], labels_train], axis=1)
    val_input = pd.concat([val[feature_cols[:-1]], labels_val], axis=1)
    test_input = pd.concat([test[feature_cols[:-1]], labels_test], axis=1)

    output_cols = ["rezistenta la insulina", "prediabet", "diabet zaharat tip 2"]

    datasets = {
        "Train": train,
        "Validation": val,
        "Test": test
    }

    for dataset_name, dataset in datasets.items():
        print(f"=== Distribuția etichetelor în setul {dataset_name} ===")
        for col in output_cols:
            counts = dataset[col].value_counts()
            zero_count = counts.get(0, 0)
            one_count = counts.get(1, 0)
            print(f"  {col}: 0 → {zero_count}, 1 → {one_count}")
        print("\n")

    y_train = train[output_cols].astype(int).values
    y_val = val[output_cols].astype(int).values
    y_test = test[output_cols].astype(int).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_input)
    X_val = scaler.transform(val_input)
    X_test = scaler.transform(test_input)

    scaler_stats = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist()
    }

    with open("models/mlp_model_diabetes/scaler_stats.json", "w") as f:
        json.dump(scaler_stats, f)

    input_dim = X_train.shape[1]
    output_dim = len(output_cols)
    model = create_diabetes_model(input_dim=input_dim, output_dim=output_dim)

    model.compile(
        optimizer="adam",
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    # class_weights_per_label = {}
    # for i, col in enumerate(output_cols):
    #     y_col = y_train[:, i]
    #     weights = compute_class_weight('balanced', classes=np.unique(y_col), y=y_col)
    #     class_weights_per_label[col] = dict(enumerate(weights))
    # print(class_weights_per_label)

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        validation_data=(X_val, y_val),
        # class_weight poate fi complicat pentru multi-label, îl poți omite sau gestiona manual cu sample_weight
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
    )
    plot_model_performance(history)

    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.4).astype(int)  # matrice binară

    for i, col in enumerate(output_cols):
        print(f"\n=== Classification Report pentru {col} ===")
        print(classification_report(y_test[:, i], y_pred_bin[:, i], zero_division=0))

        cm = confusion_matrix(y_test[:, i], y_pred_bin[:, i])
        labels = ["fără", "cu " + col]

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Matrice confuzie pentru {col}')
        plt.show()

    results = model.evaluate(X_test, y_test)
    print("\n=== Evaluare finală ===")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    # === 6. Afișează riscul estimat pentru primii 10 pacienți
    print("\n=== Risc estimat pentru primii 10 pacienți ===")
    for i in range(min(10, len(y_pred))):
        varsta = test.iloc[i]["Vârstă"]
        gen = "femeie" if test.iloc[i]["Ești "] == 1 else "bărbat"
        print(f"\nPacient {i + 1} | Vârstă: {varsta} ani | Gen: {gen}")

        for j, col in enumerate(output_cols):
            prob = y_pred[i][j]
            interpret = interpret_risk(prob)
            print(f"→ {col}: {prob * 100:.1f}% ({interpret})")

    # # Selectează doar coloanele numerice
    # numeric_data = train.select_dtypes(include=['number'])
    #
    # # Calculează matricea de corelație
    # correlation_matrix = numeric_data.corr()
    #
    # # Afișează heatmap-ul
    # plt.figure(figsize=(14, 10))  # dimensiune ajustabilă
    # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    # plt.title("Matricea de corelație între variabile numerice")
    # plt.show()

    model.save("models/mlp_model_diabetes/trained_diabetes_model.h5")
