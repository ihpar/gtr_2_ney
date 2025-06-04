import librosa
import numpy as np
from constants import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class PhaseDataset(Dataset):
    def __init__(self, gtr_paths, ney_paths, min_max):
        super().__init__()
        self.gtr_wav_file_paths = gtr_paths
        self.ney_wav_file_paths = ney_paths
        self.min_max = min_max

        self.X_db, self.X_phase = self._build_db_phase_list(
            self.gtr_wav_file_paths, min_max["gtr"])

        _, self.Y_phase = self._build_db_phase_list(
            self.ney_wav_file_paths, min_max["ney"])

    def _build_db_phase_list(self, file_paths, min_max):
        min_db, max_db = min_max["min"]["db"], min_max["max"]["db"]
        min_phase, max_phase = -np.pi, np.pi
        db_list = []
        phase_list = []
        for file_path in file_paths:
            signal, _ = librosa.load(file_path, mono=True, sr=SR)
            stft = librosa.stft(signal, n_fft=N_FFT,
                                hop_length=HOP)
            db = librosa.amplitude_to_db(np.abs(stft))
            db = (db - min_db) / (max_db - min_db)
            phase = np.angle(stft)
            phase = (phase - min_phase) / (max_phase - min_phase)
            db_list.append(np.expand_dims(db, axis=0))
            phase_list.append(np.expand_dims(phase, axis=0))
        return np.array(db_list, dtype=np.float32), np.array(phase_list, dtype=np.float32)

    def __getitem__(self, index):
        return self.X_db[index], self.X_phase[index], self.Y_phase[index], \
            self.gtr_wav_file_paths[index], self.ney_wav_file_paths[index]

    def __len__(self):
        return len(self.gtr_wav_file_paths)


def build_wav_paths_list(dir_path):
    wav_files = sorted([f.stem for f in Path(dir_path).rglob("*.wav")],
                       key=lambda x: int(x.split("_")[1]))

    return [dir_path + stem + ".wav" for stem in wav_files]


def build_phase_data_loaders(min_max, test_size=0.2):
    gtr_wav_file_paths = build_wav_paths_list(GTR_AUDIO_FEATURES_DIR)
    ney_wav_file_paths = build_wav_paths_list(NEY_AUDIO_FEATURES_DIR)

    if test_size > 0:
        X_train, X_test, Y_train, Y_test = train_test_split(
            gtr_wav_file_paths,
            ney_wav_file_paths,
            test_size=test_size,
            random_state=42)
    else:
        X_train = gtr_wav_file_paths
        Y_train = ney_wav_file_paths
        X_test, Y_test = None, None

    audio_dataset_train = PhaseDataset(X_train, Y_train, min_max)
    audio_dataloader_train = DataLoader(audio_dataset_train,
                                        batch_size=8,
                                        shuffle=True,
                                        drop_last=True)
    audio_dataloader_test = None
    if X_test is not None:
        audio_dataset_test = PhaseDataset(X_test, Y_test, min_max)
        audio_dataloader_test = DataLoader(audio_dataset_test,
                                           batch_size=8,
                                           shuffle=False,
                                           drop_last=True)

    return audio_dataloader_train, audio_dataloader_test


if __name__ == "__main__":
    import pickle

    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)
    dl_train, dl_test = build_phase_data_loaders(min_max, test_size=0.05)
    x_db, x_phase, y_phase, xp, yp = next(iter(dl_train))
    print(x_db.shape)
