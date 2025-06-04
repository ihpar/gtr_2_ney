import torch
import librosa
import numpy as np
import torch.utils
import torch.utils.data
from constants import *
from utils import stitch_wave_chunks


def predict_polar(model: torch.nn.Module,
                  data_loader,
                  mini,
                  maxi,
                  limit=10,
                  from_db=False
                  ):

    model.eval()
    predictions, targets = None, None

    i = 0
    x_paths: list[str]
    y_paths: list[str]
    for x, y, x_paths, y_paths in data_loader:
        print(", ".join([p.replace("dataset/features/gtr/", "")
              for p in x_paths]))
        print(", ".join([p.replace("dataset/features/ney/", "")
              for p in y_paths]))

        with torch.no_grad():
            pred = model(x).numpy().squeeze(axis=1)

        y = y.numpy().squeeze(axis=1)

        pred = pred * (maxi - mini) + mini
        target = y * (maxi - mini) + mini
        if from_db:
            pred = librosa.db_to_amplitude(pred)
            target = librosa.db_to_amplitude(target)

        if predictions is None:
            predictions = np.copy(pred)
            targets = np.copy(target)
        else:
            predictions = np.concatenate(
                (predictions, pred), axis=0)
            targets = np.concatenate((targets, target), axis=0)

        print("-" * 50)

    return predictions, targets


def predict_phase(model: torch.nn.Module,
                  data_loader,
                  limit=10):
    model.eval()
    predictions, targets = None, None
    i = 0
    for X_db, X_phase, Y_phase, x_paths, y_paths in data_loader:
        print(", ".join([p.replace("dataset/features/gtr/", "")
                         for p in x_paths]))
        print(", ".join([p.replace("dataset/features/ney/", "")
                         for p in y_paths]))

        with torch.no_grad():
            pred = model(torch.cat((X_db, X_phase), dim=1)
                         ).numpy().squeeze(axis=1)
        Y_phase = Y_phase.numpy().squeeze(axis=1)

        if predictions is None:
            predictions = np.copy(pred)
            targets = np.copy(Y_phase)
        else:
            predictions = np.concatenate(
                (predictions, pred), axis=0)
            targets = np.concatenate((targets, Y_phase), axis=0)
        print("-" * 50)
        i += X_db.size()[0]
        if i >= limit:
            break
    return predictions, targets


def get_phases(data_loader,
               instrument="ney",
               limit=10):
    i = 0
    phases = None
    for x, y, x_paths, y_paths in data_loader:

        if instrument == "gtr":
            phase = x.numpy().squeeze(axis=1)
            print(", ".join([p.replace("dataset/features/gtr/", "")
                             for p in x_paths]))
        else:
            phase = y.numpy().squeeze(axis=1)
            print(", ".join([p.replace("dataset/features/ney/", "")
                             for p in y_paths]))

        phase = phase * 2.0 * np.pi - np.pi  # [0, 1] -> [-pi, pi]
        if phases is None:
            phases = np.copy(phase)
        else:
            phases = np.concatenate((phases, phase), axis=0)

        print("-" * 50)
        i += x.size()[0]
        if i >= limit:
            break
    return phases


def make_wav(magnitudes, phases):
    wave_chunks = []
    for chunk_0, chunk_1 in zip(magnitudes, phases):
        chunk = chunk_0 * np.exp(1j * chunk_1)
        wave_chunk = librosa.istft(chunk, n_fft=N_FFT, hop_length=HOP)
        wave_chunks.append(wave_chunk)

    stitched_wave = stitch_wave_chunks(wave_chunks)
    return stitched_wave


if __name__ == "__main__":
    import pickle
    from models.model_0 import Model_0
    from dataset import build_data_loaders
    part = "magnitude"
    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)
        print(min_max)

    _, test_data_loader = build_data_loaders(
        min_max, part=part, test_size=0.1)

    model = Model_0()
    predictions, targets = predict_polar(model,
                                         test_data_loader,
                                         min_max["ney"]["min"][part],
                                         min_max["ney"]["max"][part],
                                         limit=12)
    print(predictions.shape, targets.shape)

    _, test_data_loader = build_data_loaders(
        min_max, part="phase", test_size=0.1)
    phases = get_phases(test_data_loader, instrument="ney", limit=12)
    print(phases.shape)
