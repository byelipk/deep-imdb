import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense

(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(nb_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    # create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# Vectorized training data
x_train = vectorize_sequences(train_data)
y_train = np.asarray(train_labels).astype('float32')

# Vectorized labels
x_test = vectorize_sequences(test_data)
y_test = np.asarray(test_labels).astype('float32')

# Save test data
np.save("imdb_x_test.npy", x_test)
np.save("imdb_y_test.npy", y_test)


# The control network
control_model = Sequential()
control_model.add(Dense(16, activation='relu', input_dim=10000))
control_model.add(Dense(16, activation='relu'))
control_model.add(Dense(1, activation='sigmoid'))

# Fighting overfitting 1
#
# Reduce network size
netsize_model = Sequential()
netsize_model.add(Dense(4, activation='relu', input_dim=10000))
netsize_model.add(Dense(4, activation='relu'))
netsize_model.add(Dense(1, activation='sigmoid'))

# Fighting overfitting 2
#
# Adding weight regularization
from keras.regularizers import l2
l2_model = Sequential()
l2_model.add(
    Dense(16, W_regularizer=l2(0.001), activation='relu', input_dim=10000))
l2_model.add(
    Dense(16, W_regularizer=l2(0.001), activation='relu'))
l2_model.add(Dense(1, activation='sigmoid'))

# Fighting overfitting 3
#
# Adding dropout
from keras.layers import Dropout
dropout_model = Sequential()
dropout_model.add(Dense(16, activation='relu', input_dim=10000))
dropout_model.add(Dropout(0.5))
dropout_model.add(Dense(16, activation='relu'))
dropout_model.add(Dropout(0.5))
dropout_model.add(Dense(1, activation='sigmoid'))

# Fighting overfitting 4
#
# Adding all three tactics
from keras.layers import Dropout
all_model = Sequential()
all_model.add(
    Dense(4, W_regularizer=l2(0.001), activation='relu', input_dim=10000))
all_model.add(Dropout(0.5))
all_model.add(
    Dense(4, W_regularizer=l2(0.001), activation='relu'))
all_model.add(Dropout(0.5))
all_model.add(Dense(1, activation='sigmoid'))


models = [
    {"model": control_model,  "name": "Control"},
    {"model": netsize_model,  "name": "Reduced Network Size"},
    {"model": l2_model,       "name": "L2 Regularization"},
    {"model": dropout_model,  "name": "Dropout"},
    {"model": all_model,  "name": "All"},
]

# Validation data
x_val = x_train[:10000]
y_val = y_train[:10000]
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]

histories = []
nb_epoch  = 20
batch_size = 512

for model_dict in models:
    name = model_dict["name"]

    print(name, "Model: Compiling")
    model_dict["model"].compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',  # This is a binary classification problem
        metrics=['accuracy'])

    print(name, "Model: Training")
    h = model_dict["model"].fit(
        partial_x_train,
        partial_y_train,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_val, y_val))

    print(name, "Model: Saving")
    filename = name
    filename = filename.lower()
    filename = filename.replace(" ", "_")
    filename = filename + ".h5"
    model_dict["model"].save(filename)

    print(name, "Model: Appending cross validation history\n")
    histories.append(h.history)

print("Saving cross validation history for models...")
import json
with open("imdb_histories", "w") as f:
    json.dump(histories, f)
