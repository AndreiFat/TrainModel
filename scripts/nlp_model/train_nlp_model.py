import ast
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from scripts.nlp_model.create_nlp_model import create_nlp_model
from scripts.utils.load_data.load_data_utils import load_data


# --- Preprocesare text ---
def preprocess_text(df):
    return (
        df["Ce alte simptome sau boli prezinți?"].fillna(''))


def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    # Pierdere (loss)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Pierdere antrenare')
    plt.plot(history.history['val_loss'], label='Pierdere validare')
    plt.title("Evoluția pierderii (loss)")
    plt.xlabel("Epoci")
    plt.ylabel("Pierdere")
    plt.legend()

    # Precizie
    plt.subplot(1, 2, 2)
    plt.plot(history.history['precision'], label='Precizie antrenare')
    plt.plot(history.history['val_precision'], label='Precizie validare')
    plt.title("Evoluția preciziei")
    plt.xlabel("Epoci")
    plt.ylabel("Precizie")
    plt.legend()

    plt.tight_layout()
    plt.show()


def print_classification_metrics(y_true, y_pred, class_names):
    print("\nRaport clasificare per etichetă:\n")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)


def plot_confusion_per_class(y_true, y_pred, class_names):
    correct = (y_true & y_pred).sum(axis=0)
    total_true = y_true.sum(axis=0)
    total_pred = y_pred.sum(axis=0)

    recall = correct / np.clip(total_true, 1, None)
    precision = correct / np.clip(total_pred, 1, None)

    plt.figure(figsize=(14, 5))
    x = range(len(class_names))
    plt.bar(x, precision, alpha=0.6, label='Precizie')
    plt.bar(x, recall, alpha=0.6, label='Recall')
    plt.xticks(x, class_names, rotation=90)
    plt.title("Precizie și recall pe fiecare etichetă")
    plt.xlabel("Etichete")
    plt.ylabel("Valori")
    plt.legend()
    plt.tight_layout()
    plt.show()


def train_model():
    # --- Încarcă datele ---
    train_df = load_data("data/datasets/train/train_nlp.csv", sep=';')
    val_df = load_data("data/datasets/validation/val_nlp.csv", sep=';')
    test_df = load_data("data/datasets/test/test_nlp.csv", sep=';')

    train_texts = preprocess_text(train_df).tolist()
    val_texts = preprocess_text(val_df).tolist()
    test_texts = preprocess_text(test_df).tolist()

    # --- Extrage etichetele multi-label cu MultiLabelBinarizer ---
    def extract_labels(df, mlb=None):
        labels_parsed = df['labels'].apply(ast.literal_eval)
        if mlb is None:
            mlb = MultiLabelBinarizer()
            y = mlb.fit_transform(labels_parsed)
            return y, mlb
        else:
            y = mlb.transform(labels_parsed)
            return y

    y_train, mlb = extract_labels(train_df)
    y_val = extract_labels(val_df, mlb)
    y_test = extract_labels(test_df, mlb)

    print("Clase etichete:", mlb.classes_)

    # --- Tokenizare ---
    max_words = 100000
    max_len = 100

    all_texts = train_texts + val_texts + test_texts
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_texts)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding='post')
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=max_len, padding='post')
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_texts), maxlen=max_len, padding='post')

    # --- Creează model ---
    num_labels = y_train.shape[1]
    model = create_nlp_model(max_words, max_len, num_labels)

    # --- Antrenare ---
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=callbacks
    )

    # --- Evaluare test ---
    # După antrenament:
    test_metrics = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_metrics[0]:.4f}")
    print(f"Test Accuracy: {test_metrics[1]:.4f}")
    print(f"Test AUC: {test_metrics[2]:.4f}")
    print(f"Test Precision: {test_metrics[3]:.4f}")
    print(f"Test Recall: {test_metrics[4]:.4f}")

    # --- Predicție exemplu ---
    sample_text = ["durere abdominala si oboseala cronica", "am tiroidita hashimoto si infarct miocardic",
                   "am pofta de dulce"]

    # preprocesare text
    sample_seq = pad_sequences(tokenizer.texts_to_sequences(sample_text), maxlen=max_len, padding='post')

    # predicție
    pred = model.predict(sample_seq)

    # conversie la etichete binare
    pred_labels_bin = (pred > 0.5).astype(int)

    # transformare în clase (folosind MultiLabelBinarizer)
    pred_classes = mlb.inverse_transform(pred_labels_bin)

    # afișare
    for i, text in enumerate(sample_text):
        print(f"\nText: {text}")
        print("Etichete prezise:", pred_classes[i])

    sample_text = [
        "tiroida",
        "tiroida hashimoto",
        "hashimoto",
        "oboseala",
        "palpitatii",
        "infarct",
        "ficat gras",
        "pofta de dulce"
    ]

    sample_seq = pad_sequences(tokenizer.texts_to_sequences(sample_text), maxlen=max_len, padding='post')
    pred = model.predict(sample_seq)

    for i, text in enumerate(sample_text):
        print(f"\nText: {text}")
        for label, prob in zip(mlb.classes_, pred[i]):
            if prob > 0.5:
                print(f"  {label}: {prob:.2f}")

    model.save("models/nlp_model/new_model/nlp_model.h5")
    # Salvare tokenizer
    with open("models/nlp_model/new_model/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    # Salvare mlb (MultiLabelBinarizer)
    with open("models/nlp_model/new_model/mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)

    plot_training_history(history)
