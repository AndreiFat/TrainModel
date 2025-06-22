from tensorflow.keras.layers import Dense, Input, BatchNormalization, ELU
from tensorflow.keras.models import Sequential


def create_diabetes_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128),
        ELU(alpha=1.0),
        BatchNormalization(),

        Dense(64),
        ELU(alpha=1.0),
        BatchNormalization(),

        Dense(32),
        ELU(alpha=1.0),
        BatchNormalization(),

        Dense(output_dim, activation='sigmoid')
    ])
    return model
