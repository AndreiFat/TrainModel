from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense, \
    SpatialDropout1D, LayerNormalization, GlobalMaxPooling1D
from tensorflow.keras.regularizers import l2


def create_nlp_model(max_len, max_vocab, num_classes):
    model = Sequential([
        Embedding(input_dim=max_vocab, output_dim=128, input_length=max_len),
        SpatialDropout1D(0.5),
        Bidirectional(LSTM(64, return_sequences=True)),
        LayerNormalization(),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(len(num_classes.classes_), activation='sigmoid')
    ])
    return model
