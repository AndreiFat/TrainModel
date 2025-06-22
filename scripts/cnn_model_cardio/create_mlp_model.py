from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def create_mlp_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(265, activation='relu', kernel_regularizer=l2(0.001)),  # am crescut de la 256 la 512
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),  # am crescut de la 128 la 256
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation='relu'),  # am crescut de la 64 la 128
        BatchNormalization(),
        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])
    return model
