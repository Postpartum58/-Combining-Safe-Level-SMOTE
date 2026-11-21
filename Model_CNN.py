import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from Evaluation import evaluation


def Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, epochs):

    Train_Data = Train_Data.reshape(Train_Data.shape[0], Train_Data.shape[1], 1)
    Test_Data = Test_Data.reshape(Test_Data.shape[0], Test_Data.shape[1], 1)

    num_features = Train_Data.shape[1]
    output_dim = Train_Target.shape[1]

    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(num_features, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    model.fit(Train_Data, Train_Target, epochs=epochs, batch_size=32, verbose=1)

    pred = model.predict(Test_Data)
    pred_classes = np.argmax(pred, axis=1)
    true_classes = np.argmax(Test_Target, axis=1)
    Eval = evaluation(true_classes, pred_classes)
    return Eval, pred_classes


