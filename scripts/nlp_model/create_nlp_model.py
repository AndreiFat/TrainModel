from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def create_nlp_model(max_len, max_vocab, num_classes):
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=128, input_length=max_len),

        # Recurrent Layers
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(GRU(32, return_sequences=True)),

        GlobalMaxPooling1D(),

        # Normalization before Dense
        BatchNormalization(),

        # Dense Layers
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),

        Dense(num_classes, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            BinaryAccuracy(name='binary_accuracy'),
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
    )

    return model
