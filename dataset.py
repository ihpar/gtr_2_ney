import pickle
import numpy as np
from constants import *
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class FeatureDataset(Dataset):
    def __init__(self, x_dirs, y_dirs, min_max, part):
        self.x_dirs = x_dirs
        self.y_dirs = y_dirs
        self.min_max = min_max
        self.part = part

        # gtr files
        self.x, self.file_paths_x = self._build_feature_data(
            self.x_dirs,
            self.min_max["gtr"],
            self.part
        )

        # ney files
        self.y, self.file_paths_y = self._build_feature_data(
            self.y_dirs,
            self.min_max["ney"],
            self.part
        )

    def _build_feature_data(self, dir_paths, min_max, part):
        features_arr = []
        paths_arr = []
        for dir_path in dir_paths:
            features, file_paths = self._get_features_in_dir(
                dir_path,
                min_max,
                part
            )
            features_arr.extend(features)
            paths_arr.extend(file_paths)

        features_arr = np.array(features_arr, dtype=np.float32)

        paths_arr = np.array(paths_arr)
        return features_arr, paths_arr

    def _get_features_in_dir(self, dir_path, min_max, part):
        if part not in ["magnitude", "phase", "db"]:
            raise Exception(f"Invalid part name: {part}")

        features = []
        file_paths = []
        sorted_files = sorted(list(Path(dir_path).iterdir()),
                              key=lambda x: int(x.stem.split("_")[1]))
        for f in sorted_files:
            with open(f, "rb") as handle:
                chunk = pickle.load(handle)

            data = chunk[part]
            mini = min_max["min"][part]
            maxi = min_max["max"][part]
            data = (data - mini) / (maxi - mini)
            features.append(np.expand_dims(data, axis=0))

            file_paths.append(str(f))
        return features, file_paths

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.file_paths_x[index], self.file_paths_y[index]

    def __len__(self):
        return self.x.shape[0]


def build_data_loaders(min_max, part=None, test_size=0.2, test_only=False):
    gtr_feature_dirs = sorted(
        [f for f in Path(GTR_FEATURE_DIR).iterdir() if f.is_dir()])

    gtr_feature_dirs = gtr_feature_dirs + gtr_feature_dirs

    ney_feature_dirs = sorted(
        [f for f in Path(NEY_FEATURE_DIR).iterdir() if f.is_dir()])

    ney_feature_dirs = ney_feature_dirs + \
        ney_feature_dirs[30:] + ney_feature_dirs[:30]

    train_dataset, train_data_loader = None, None
    test_dataset, test_data_loader = None, None

    if test_size > 0:
        x_train_dirs, x_test_dirs, y_train_dirs, y_test_dirs = train_test_split(
            gtr_feature_dirs,
            ney_feature_dirs,
            test_size=test_size,
            random_state=42)
        test_dataset = FeatureDataset(x_test_dirs, y_test_dirs,
                                      min_max, part)
        test_data_loader = DataLoader(dataset=test_dataset,
                                      batch_size=8,
                                      shuffle=False,
                                      drop_last=True)
    else:
        x_train_dirs = gtr_feature_dirs
        y_train_dirs = ney_feature_dirs

    if not test_only:
        train_dataset = FeatureDataset(x_train_dirs, y_train_dirs,
                                       min_max, part)

        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=8,
                                       shuffle=True,
                                       drop_last=True)

    return train_data_loader, test_data_loader


if __name__ == "__main__":
    from plotter import plot_heatmaps
    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)

    train_loader, test_loader = build_data_loaders(min_max,
                                                   part="db",
                                                   test_size=0.1)

    x, y, _, _ = next(iter(train_loader))
    print(x.size())
    print(y.size())

    plot_heatmaps(x[0].numpy().squeeze(axis=0),
                  y[0].numpy().squeeze(axis=0))
