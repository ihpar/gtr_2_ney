import torch
import librosa
import numpy as np
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
