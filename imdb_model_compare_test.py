import numpy as np
from keras.models import load_model


X_test = np.load("imdb_x_test.npy")
y_test = np.load("imdb_y_test.npy")

filenames = [
    "all.h5",
    "control.h5",
    "dropout.h5",
    "l2_regularization.h5",
    "reduced_network_size.h5"
]

for filename in filenames:
    model   = load_model(filename)
    results = model.evaluate(X_test, y_test)
    print()
    print(filename)
    print("Test loss:", results[0])
    print("Model Accuracy:", results[1])
    print()
