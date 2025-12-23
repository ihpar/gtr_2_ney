import torch
import librosa
import numpy as np
from constants import *
import numpy.typing as npt


class Inferer:
    def __init__(self, device, mini, maxi, len_window, len_overlap, n_fft, hop):
        self.device = device
        self.mini = mini
        self.maxi = maxi
        self.len_window = len_window
        self.len_overlap = len_overlap
        self.n_fft = n_fft
        self.hop = hop
        self.start_offset = len_window - len_overlap

    def _get_X(self, signal: npt.NDArray, scale_phase=False):
        i = 0
        dbs = []
        phases = []
        while True:
            start = i * self.start_offset
            end = start + self.len_window
            window = signal[start:end]
            # pad with trailing 0's if window is smaller than required length
            if len(window) < self.len_window:
                window = np.pad(window, (0, self.len_window - len(window)))

            # stft -> phase, magnitude & db
            stft = librosa.stft(window, n_fft=self.n_fft, hop_length=self.hop)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            if scale_phase:
                phase = (phase + np.pi) / (2 * np.pi)
            phases.append(phase)
            db = librosa.amplitude_to_db(magnitude)
            # normalization
            db = (db - self.mini) / (self.maxi - self.mini)
            db = np.expand_dims(db, axis=0)
            dbs.append(db)

            if end >= len(signal):
                break
            i += 1

        dbs = torch.from_numpy(np.array(dbs, dtype=np.float32))
        return dbs, phases

    def _get_predictions(self, model, dbs, verbose=False, as_db=False):
        if verbose:
            print("Getting magnitude predictions...")

        predictions = []
        num_dbs = dbs.size()[0]
        with torch.no_grad():
            for i in range(num_dbs):
                y = dbs[i:i+1].to(self.device)
                y_hat = (model(y)).to(torch.device("cpu"))
                y_hat = y_hat[0].numpy().squeeze(axis=0)
                if not as_db:
                    y_hat = y_hat * (self.maxi - self.mini) + self.mini
                    y_hat = librosa.db_to_amplitude(y_hat)
                predictions.append(y_hat)
                if verbose:
                    print(f"  magnitude {i+1}/{num_dbs}")
        return predictions

    def _inverse_preds(self, predictions, verbose, phases=None):
        if verbose:
            print("Inverting predictions...")
        inverses = []
        for i, prediction in enumerate(predictions):
            if phases is None:
                inverse = librosa.griffinlim(prediction, n_iter=128,
                                             n_fft=self.n_fft,
                                             hop_length=self.hop,
                                             random_state=42)
            else:
                inverse = librosa.istft(prediction * np.exp(1j * phases[i]),
                                        n_fft=N_FFT, hop_length=HOP)
            inverses.append(inverse)
        return inverses

    def _stitch(self, inverses, verbose):
        if verbose:
            print("Stitching predictions...")
        stitched_wave = np.array([], dtype=np.float32)
        fade_in = np.linspace(0, 1, self.len_overlap)
        fade_out = np.linspace(1, 0, self.len_overlap)

        for i, inverse in enumerate(inverses):
            if i > 0:
                inverse[0:self.len_overlap] *= fade_in

            if len(stitched_wave) > 0:
                stitched_wave[-self.len_overlap:] += inverse[:self.len_overlap]
                stitched_wave = np.append(
                    stitched_wave, inverse[self.len_overlap:])
            else:
                stitched_wave = np.append(stitched_wave, inverse)

            # fade out all wave chunks
            stitched_wave[-self.len_overlap:] *= fade_out

        stitched_wave /= (np.max(np.abs(stitched_wave)) + 0.001)
        return stitched_wave

    def infer(self, model: torch.nn.Module, signal, use_gl=False, verbose=True):
        signal /= np.max(np.abs(signal))
        X, phases = self._get_X(signal)
        predictions = self._get_predictions(model, X, verbose)
        if use_gl:
            inverses = self._inverse_preds(predictions, verbose)
        else:
            inverses = self._inverse_preds(predictions, verbose, phases)

        return self._stitch(inverses, verbose)
