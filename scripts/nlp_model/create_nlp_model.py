from keras.src.layers import GRU
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, \
    SpatialDropout1D, Conv1D, LayerNormalization, GlobalMaxPooling1D


def create_nlp_model(max_len, max_vocab, num_classes):
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=256, input_length=max_len),
        SpatialDropout1D(0.5),
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        Bidirectional(LSTM(64, return_sequences=True)),
        LayerNormalization(),
        Bidirectional(GRU(32, return_sequences=True)),  # output 3D
        GlobalMaxPooling1D(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(num_classes.classes_), activation='sigmoid')
    ])
    return model
