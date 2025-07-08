from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2


def create_diabetes_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(),

        Dense(128, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(),

        Dense(64, kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        LeakyReLU(),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])
    return model
