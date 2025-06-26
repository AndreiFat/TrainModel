import ast
import time

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from models.F1Score import F1Score
from models.TimeHistory import TimeHistory
from scripts.nlp_model.create_nlp_model import create_nlp_model
from scripts.utils.load_data.load_data_utils import load_data
from scripts.utils.plots.plotNLP import plot_history, plot_confusion_matrices
from scripts.utils.test.test_text_nlp import test_text_nlp


def concat_text(df, text_cols):
    if isinstance(text_cols, str):
        return df[text_cols].fillna("")
    else:
        return df[text_cols].fillna("").agg(" ".join, axis=1)


def train_model():
    max_len = 300
    max_vocab = 16000

    # 1. Încarcă datele
    train = load_data("data/datasets/train/train.csv", sep=';')
    val = load_data("data/datasets/validation/val.csv", sep=';')
    test = load_data("data/datasets/test/test.csv", sep=';')

    text_cols = [
        'Ce alte simptome sau boli prezinți?',
        'In prezent, care este cea mai mare provocare a ta? Ce crezi ca te împiedica sa slăbești si sa ai o stare buna de sănătate? ',
        # 'Ce te-a împiedicat in trecut sa slăbești? De ce ai eșuat la alte încercări? '
    ]

    for col in text_cols:
        train[col] = train[col].fillna("").astype(str)
        val[col] = val[col].fillna("").astype(str)
        test[col] = test[col].fillna("").astype(str)

    train["labels"] = train["labels"].apply(ast.literal_eval)
    val["labels"] = val["labels"].apply(ast.literal_eval)
    test["labels"] = test["labels"].apply(ast.literal_eval)

    train_text = concat_text(train, text_cols)
    val_text = concat_text(val, text_cols)
    test_text = concat_text(test, text_cols)

    # 2. Tokenizează textul
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_text)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=max_len)
    X_val = pad_sequences(tokenizer.texts_to_sequences(val_text), maxlen=max_len)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=max_len)

    # 3. Transformă etichetele în one-hot vectori
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(train["labels"])
    y_val = mlb.transform(val["labels"])
    y_test = mlb.transform(test["labels"])

    print("Etichete:", mlb.classes_)

    # === Creează și compilează modelul ===
    model = create_nlp_model(max_len=max_len, max_vocab=max_vocab, num_classes=mlb)

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    time_callback = TimeHistory()

    model.compile(
        loss='binary_crossentropy',
        optimizer="adam",
        metrics=[
            "accuracy",
            BinaryAccuracy(name='binary_accuracy'),
            Precision(name='precision'),
            Recall(name='recall'),
            AUC(name='auc'),
            F1Score(name='f1_score')
        ]
    )

    # === Antrenare ===
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr, time_callback]
    )

    plot_history(history)
    plot_confusion_matrices(model, X_test, y_test, class_names=mlb.classes_, threshold=0.5)

    # === Timp antrenare ===
    print("⏱️ Timp per epocă:", time_callback.times)
    print("⏱️ Timp total antrenare:", sum(time_callback.times))

    # === Plotare evoluție metrici ===
    plt.plot(history.history['precision'], label='Train Precision')
    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.title("Evoluție Precision")
    plt.xlabel("Epocă")
    plt.ylabel("Precizie")
    plt.legend()
    plt.show()

    # === Evaluare ===
    results = model.evaluate(X_test, y_test)
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    # === Classification report ===
    y_pred = model.predict(X_test)
    y_pred_bin = (y_pred > 0.5).astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_bin, target_names=mlb.classes_, zero_division=0))

    # === Funcție de predicție ===
    def predict_labels(test_sample):
        seq = pad_sequences(tokenizer.texts_to_sequences([test_sample]), maxlen=max_len)
        pred = model.predict(seq)[0]
        threshold = 0.5
        labels = [mlb.classes_[i] for i, p in enumerate(pred) if p > threshold]
        return labels

    test_texts = test_text_nlp()

    # Timp de inferență pe textele demo
    inference_times = []
    for text in test_texts:
        start_time = time.time()
        labels = predict_labels(text)
        duration = time.time() - start_time
        inference_times.append(duration)

        print(f"Text: {text}")
        print(f"Labels prezise: {labels}")
        print(f"Timp predicție: {duration:.4f} sec\n")

    print(f"\nTimp mediu predicție: {sum(inference_times) / len(inference_times):.4f} sec")
    print(f"Timp total pentru {len(test_texts)} texte: {sum(inference_times):.2f} sec")
