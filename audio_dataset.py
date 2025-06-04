import librosa
import numpy as np
from constants import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class AudioDataset(Dataset):
    def __init__(self, gtr_paths, ney_paths):
        super().__init__()
        self.gtr_wav_file_paths = gtr_paths
        self.ney_wav_file_paths = ney_paths
        self.X = self.build_wav_data_list(
            self.gtr_wav_file_paths)
        self.Y = self.build_wav_data_list(
            self.ney_wav_file_paths)

    def build_wav_data_list(self, file_paths):
        wav_data_list = []
        for file_path in file_paths:
            signal, _ = librosa.load(file_path, mono=True, sr=SR)
            wav_data_list.append(np.expand_dims(signal, axis=0))
        return np.array(wav_data_list, dtype=np.float32)

    def __getitem__(self, index):
        return self.X[index], self.Y[index], \
            self.gtr_wav_file_paths[index], self.ney_wav_file_paths[index]

    def __len__(self):
        return len(self.gtr_wav_file_paths)


def build_wav_paths_list(dir_path):
    wav_files = sorted([f.stem for f in Path(dir_path).rglob("*.wav")],
                       key=lambda x: int(x.split("_")[1]))

    return [dir_path + stem + ".wav" for stem in wav_files]


def build_audio_data_loaders(test_size=0.2):
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

    audio_dataset_train = AudioDataset(X_train, Y_train)
    audio_dataloader_train = DataLoader(audio_dataset_train,
                                        batch_size=8,
                                        shuffle=True,
                                        drop_last=True)
    audio_dataloader_test = None
    if X_test is not None:
        audio_dataset_test = AudioDataset(X_test, Y_test)
        audio_dataloader_test = DataLoader(audio_dataset_test,
                                           batch_size=8,
                                           shuffle=False,
                                           drop_last=True)

    return audio_dataloader_train, audio_dataloader_test


if __name__ == "__main__":
    dl_train, dl_test = build_audio_data_loaders(test_size=0.05)
    x, y, xp, yp = next(iter(dl_train))
    signal_x = x[0]
    path_x = xp[0]
    signal_y = y[0]
    path_y = yp[0]
    signal_x = signal_x.numpy().squeeze(axis=0)
    signal_y = signal_y.numpy().squeeze(axis=0)
    print(path_x, path_y, signal_x.shape, signal_y.shape)
