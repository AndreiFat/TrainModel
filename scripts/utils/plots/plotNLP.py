import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def plot_history(history):
    metrics = ['loss', 'val_loss', 'binary_accuracy', 'val_binary_accuracy',
               'precision', 'val_precision', 'recall', 'val_recall',
               'auc', 'val_auc']

    for metric in metrics:
        if metric in history.history:
            plt.figure(figsize=(6, 4))
            plt.plot(history.history[metric], label=metric)
            plt.title(metric)
            plt.xlabel("Epocă")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.show()


def plot_training_history(history):
    """
    Afișează o diagramă cu evoluția metricalor în timpul antrenării: pierdere (loss) și F1-score pentru seturile de antrenare și validare.

    Parametru:
    - history: obiectul returnat de model.fit(), conține istoricul antrenării.
    """
    # Extragem metricele din istoric
    loss = history.history['loss']
    val_loss = history.history.get('val_loss')

    f1 = history.history.get('f1_score') or history.history.get('f1')  # poate fi numită f1 sau f1_score
    val_f1 = history.history.get('val_f1_score') or history.history.get('val_f1')

    epochs = range(1, len(loss) + 1)

    # Cream figura cu două subploturi: Loss și F1-score
    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b-', label='Loss (train)')
    if val_loss:
        plt.plot(epochs, val_loss, 'r--', label='Loss (val)')
    plt.title('Evoluția pierderii (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Subplot 2: F1-score
    plt.subplot(1, 2, 2)
    if f1:
        plt.plot(epochs, f1, 'b-', label='F1-score (train)')
    if val_f1:
        plt.plot(epochs, val_f1, 'r--', label='F1-score (val)')
    plt.title('Evoluția scorului F1')
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.legend()

    # Afișăm totul
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(model, x_val, y_val, class_names=None, threshold=0.5):
    """
    Afișează matricea de confuzie pentru fiecare clasă într-un task multi-label.

    Args:
        model: modelul antrenat (Keras).
        x_val: datele de validare (ex. X_test).
        y_val: etichetele reale binare (forma: [samples, num_classes]).
        class_names: lista cu numele claselor (dacă nu, se folosește indexul).
        threshold: pragul pentru binarizare (default = 0.5).
    """
    # Predicții pe setul de validare
    y_pred = model.predict(x_val)
    y_pred_bin = (y_pred >= threshold).astype(int)

    num_classes = y_val.shape[1]
    if class_names is None:
        class_names = [f"Clasa {i}" for i in range(num_classes)]

    for i in range(num_classes):
        cm = confusion_matrix(y_val[:, i], y_pred_bin[:, i])
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Matricea de confuzie - {class_names[i]}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()

        print(f"\nClasificare pentru {class_names[i]}:\n")
        print(classification_report(y_val[:, i], y_pred_bin[:, i], digits=3))
