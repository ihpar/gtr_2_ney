import torch
import pickle
import librosa
import numpy as np
import soundfile as sf
from constants import *
import numpy.typing as npt
from disk_utils import load_model


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

    def _get_phases(self, model_phase, dbs, phases, verbose):
        if verbose:
            print("Getting phase predictions...")
        predictions = []
        num_phases = len(phases)
        with torch.no_grad():
            for i in range(num_phases):
                db = torch.from_numpy(
                    np.array([np.expand_dims(dbs[i], axis=0)]))
                phase = torch.from_numpy(
                    np.array([np.expand_dims(phases[i], axis=0)]))
                y = torch.cat((db, phase), dim=1).to(self.device)
                y_hat = (model_phase(y)).to(torch.device("cpu"))
                y_hat = y_hat[0].numpy().squeeze(axis=0)
                y_hat = y_hat * (2 * np.pi) - np.pi
                predictions.append(y_hat)
                if verbose:
                    print(f"  phase {i+1}/{num_phases}")
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

    def _inverse_preds_scaled(self, dbs, phases, verbose):
        if verbose:
            print("Inverting predictions phases...")

        inverses = []
        for db, phase in zip(dbs, phases):
            magnitude = db * (self.maxi - self.mini) + self.mini
            magnitude = librosa.db_to_amplitude(magnitude)
            inverse = librosa.istft(magnitude * np.exp(1j * phase),
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

    def _predict_hifi(self, model, signals, verbose=True):
        if verbose:
            print("Getting signal predictions...")
        predictions = []
        with torch.no_grad():
            for i, signal in enumerate(signals):
                x = np.array([np.expand_dims(signal, axis=0)],
                             dtype=np.float32)
                x = (torch.from_numpy(x)).to(self.device)
                y = model(x)
                y = y.to(torch.device("cpu"))
                y = (y.numpy()[0]).squeeze(axis=0)
                predictions.append(y)
                if verbose:
                    print(f"  hifi {i+1}/{len(signals)}")
        return predictions

    def infer(self, model: torch.nn.Module, signal, use_gl=False, verbose=True):
        signal /= np.max(np.abs(signal))
        X, phases = self._get_X(signal)
        predictions = self._get_predictions(model, X, verbose)
        if use_gl:
            inverses = self._inverse_preds(predictions, verbose)
        else:
            inverses = self._inverse_preds(predictions, verbose, phases)

        return self._stitch(inverses, verbose)

    def infer_phase(self, model_magnitude, model_phase, src_signal, verbose=True):
        src_signal /= np.max(np.abs(src_signal))
        X, phases = self._get_X(src_signal, scale_phase=True)
        dbs = self._get_predictions(model_magnitude, X, verbose, as_db=True)
        phases = self._get_phases(model_phase, dbs, phases, verbose)
        inverses = self._inverse_preds_scaled(dbs, phases, verbose)
        return self._stitch(inverses, verbose)

    def infer_hifi(self, model_magnitude, model_hifi, src_signal, verbose=True):
        src_signal /= np.max(np.abs(src_signal))
        X, phases = self._get_X(src_signal)
        predictions = self._get_predictions(model_magnitude, X, verbose)
        inverses = self._inverse_preds(predictions, verbose, phases)
        inverses_hifi = self._predict_hifi(model_hifi, inverses, verbose)
        return self._stitch(inverses_hifi, verbose)

    def get_magnitudes(self, model: torch.nn.Module,
                       signal_gtr: npt.NDArray,
                       signal_ney: npt.NDArray):

        signal_gtr /= np.max(np.abs(signal_gtr))
        signal_ney /= np.max(np.abs(signal_ney))
        X_gtr, _ = self._get_X(signal_gtr)
        predictions = self._get_predictions(
            model, X_gtr, verbose=False, as_db=True)

        X_ney, _ = self._get_X(signal_ney)
        return np.array(predictions), X_ney.squeeze(axis=1)


if __name__ == "__main__":
    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)

    inst = "ney"
    feature = "db"
    mini = min_max[inst]["min"][feature]
    maxi = min_max[inst]["max"][feature]
    signal, _ = librosa.load("dataset/tests/test_0.wav", mono=True, sr=SR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_magnitude = load_model("generator_sp_32_0_8_full").to(device)
    model_phase = load_model("phase_generator").to(device)
    # model_hifi = load_model("hifi_gen").to(device)

    inferer = Inferer(device, mini, maxi, WINDOW_SAMPLE_LEN,
                      OVERLAP, N_FFT, HOP)
    result = inferer.infer_phase(model_magnitude, model_phase, signal)

    # sf.write("katip_3.wav", result, SR, format="wav")
