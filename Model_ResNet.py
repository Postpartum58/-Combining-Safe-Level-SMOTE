import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from Evaluation import evaluation


def Model_ResNet(x_train, y_train, x_test, y_test, epochs=10):
    input_shape = x_train.shape[1:]

    model = Sequential([
        Conv1D(128, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(64, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(y_train.shape[1], activation='softmax')
    ])

    model.compile(
        optimizer=Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=8,
        validation_data=(x_test, y_test)
    )
    predict = model.predict(x_test)
    normalized_values = (predict - np.min(predict)) / (np.max(predict) - np.min(predict))
    pred = np.where(normalized_values >= 0.5, 1, 0)
    Eval = evaluation(y_test, pred)
    return Eval, pred


