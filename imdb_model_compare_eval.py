import json
import matplotlib.pyplot as plt

print("Loading cross validation history...")

with open("imdb_histories", "r") as f:
    histories = json.load(f)

# Visualizing the data
control  = histories[0]
ctrl_val = control["val_loss"]
epochs   = range(1, len(ctrl_val) + 1)

print("Plotting comparisions...")
for idx, hist in enumerate(histories[1:]):
    hist_val = hist["val_loss"]
    title    = "Comparison " + str(idx)

    plt.title(title)
    plt.plot(epochs, ctrl_val, 'bo')
    plt.plot(epochs, hist_val, 'b+')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.savefig("comparison" + "_" + str(idx) + ".png")
    plt.clf()
