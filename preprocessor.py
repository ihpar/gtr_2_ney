import pickle
import librosa
import numpy as np
import librosa.display
from constants import *
from pathlib import Path, PurePath


class Preprocessor:
    def __init__(self, sr=SR, n_fft=N_FFT, hop=HOP,
                 signal_hop=SIGNAL_HOP,
                 window_len=WINDOW_SAMPLE_LEN):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.signal_hop = signal_hop
        self.window_len = window_len

    def normalize_signal(self, signal):
        maxi = np.max(np.abs(signal))
        ratio = 1.0 / maxi
        signal = ratio * signal
        return signal

    def preprocess_wav_file(self, file_path):
        """
        Takes a wav file path as arg and splits into chunks of WINDOW_SAMPLE_LEN
        """
        signal, _ = librosa.load(file_path, mono=True, sr=self.sr)
        signal = self.normalize_signal(signal)

        signal_len = len(signal)
        result = {
            "file_path": file_path,
            "chunks": []
        }
        i = 0
        while True:
            start = i * self.signal_hop
            end = start + self.window_len
            if end >= signal_len:
                break

            window = signal[start: end]
            stft = librosa.stft(window, n_fft=self.n_fft,
                                hop_length=self.hop)
            # extract features
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            db = librosa.amplitude_to_db(magnitude)
            # restore stft as
            # stft = magnitude * (np.cos(phase) + 1j*np.sin(phase))
            # stft = amplitude * np.exp(1j * phase)
            result["chunks"].append({
                "start": start,
                "end": end,
                "magnitude": magnitude,
                "phase": phase,
                "db": db,
                "signal": window
            })
            i += 1

        return result

    def save_preprocessed_dataset(self, file_list, target_dir):
        """
        Processes all wav files in file_list arg.
        Saves the preprocessed files in respective directories
        """
        min_max = {
            "min": {
                "magnitude": np.inf,
                "phase": -np.pi,
                "db": np.inf,
                "signal": np.inf
            },
            "max": {
                "magnitude": -np.inf,
                "phase": np.pi,
                "db": -np.inf,
                "signal": -np.inf
            }
        }
        for file_path in file_list:
            result = self.preprocess_wav_file(file_path)
            file_stem = Path(result["file_path"]).stem
            new_dir_path = PurePath(target_dir, file_stem)
            p = Path(new_dir_path)
            p.mkdir(parents=True, exist_ok=True)
            for i, chunk in enumerate(result["chunks"]):
                min_magnitude, max_magnitude = np.min(
                    chunk["magnitude"]), np.max(chunk["magnitude"])
                min_db, max_db = np.min(
                    chunk["db"]), np.max(chunk["db"])
                min_sig, max_sig = np.min(
                    chunk["signal"]), np.max(chunk["signal"])

                if min_magnitude < min_max["min"]["magnitude"]:
                    min_max["min"]["magnitude"] = min_magnitude
                if max_magnitude > min_max["max"]["magnitude"]:
                    min_max["max"]["magnitude"] = max_magnitude

                if min_db < min_max["min"]["db"]:
                    min_max["min"]["db"] = min_db
                if max_db > min_max["max"]["db"]:
                    min_max["max"]["db"] = max_db

                if min_sig < min_max["min"]["signal"]:
                    min_max["min"]["signal"] = min_sig
                if max_sig > min_max["max"]["signal"]:
                    min_max["max"]["signal"] = max_sig

                chunk_name = f"chunk_{i}"
                chunk_path = PurePath(new_dir_path, chunk_name)
                with open(chunk_path, "wb") as handle:
                    pickle.dump(chunk, handle)

        return min_max


if __name__ == "__main__":
    pp = Preprocessor(SR, N_FFT, HOP, SIGNAL_HOP, WINDOW_SAMPLE_LEN)
    result = pp.preprocess_wav_file("dataset/ney_1/Ney_1_10.wav")
    print(len(result["chunks"]))
    print(result["chunks"][0]["magnitude"].shape)
    print(result["chunks"][0]["phase"].shape)
    print(result["chunks"][0]["db"].shape)
    print(result["chunks"][0]["signal"].shape)

    ney_wav_files = sorted(
        [data_dir + f.name for data_dir in NEY_WAV_DIRS for f in Path(data_dir).rglob("*.wav")])
    # print(ney_wav_files, "\n")

    gtr_wav_files = sorted(
        [data_dir + f.name for data_dir in GTR_WAV_DIRS for f in Path(data_dir).rglob("*.wav")])
    # print(gtr_wav_files, "\n")

    min_max_ney = pp.save_preprocessed_dataset(ney_wav_files, NEY_FEATURE_DIR)
    min_max_gtr = pp.save_preprocessed_dataset(gtr_wav_files, GTR_FEATURE_DIR)
    min_max = {
        "ney": min_max_ney,
        "gtr": min_max_gtr
    }
    print(min_max)

    with open("dataset/features/min_max.pkl", "wb") as handle:
        pickle.dump(min_max, handle)
