from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, ELU
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def create_diabetes_model(input_dim, output_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, kernel_regularizer=l2(1e-4)),
        ELU(alpha=1.0),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, kernel_regularizer=l2(1e-4)),
        ELU(alpha=1.0),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, kernel_regularizer=l2(1e-4)),
        ELU(alpha=0.0),
        BatchNormalization(),
        Dropout(0.2),

        Dense(output_dim, activation='sigmoid')
    ])
    return model
