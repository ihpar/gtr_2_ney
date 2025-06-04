import librosa
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(history, title, start=0):
    plt.figure(figsize=(8, 4))
    plt.title(title)
    train_loss = history["train"]
    val_loss = history["val"]
    epochs = np.arange(1 + start, len(train_loss) + 1)
    plt.plot(epochs, train_loss[start:], label="train")
    plt.plot(epochs, val_loss[start:], label="validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_gan_loss(history, title, start=0):
    plt.figure(figsize=(8, 4))
    plt.title(title)
    gen_loss = history["gen"]
    disc_loss = history["disc"]
    epochs = np.arange(1 + start, len(gen_loss) + 1)
    plt.plot(epochs, gen_loss[start:], label="Generator")
    plt.plot(epochs, disc_loss[start:], label="Discriminator")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_heatmaps(prediction, target, from_db=False, title=None):
    sns.set_theme(rc={"figure.figsize": (14, 5)})
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if title is not None:
        fig.suptitle(title)
    if from_db:
        prediction = librosa.amplitude_to_db(prediction)
        target = librosa.amplitude_to_db(target)
    ax1 = sns.heatmap(prediction, ax=ax1)
    ax1.set_title("Predicted")
    ax1.invert_yaxis()

    ax2 = sns.heatmap(target, ax=ax2)
    ax2.set_title("Actual")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_waves(wave_target, wave_prediction):
    fig, axs = plt.subplots(2, figsize=(8, 6))
    fig.suptitle("Target & Predicted Waves")
    axs[0].set_title("Target")
    axs[0].set_ylim([-1, 1])
    axs[0].plot(wave_target)

    axs[1].set_title("Prediction")
    axs[1].set_ylim([-1, 1])
    axs[1].plot(wave_prediction)
    fig.tight_layout()
    plt.show()
