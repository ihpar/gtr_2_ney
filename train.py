import torch
import torch.nn as nn


def train_model(model: nn.Module,
                criterion,
                optimizer,
                device,
                train_data_loader,
                test_data_loader,
                early_stop,
                num_epochs=1):

    model.to(device)
    history = {"train": [], "val": []}

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train(True)
        running_loss = 0.0
        num_train_batches = 0
        for gtr_features, ney_features, _, _ in train_data_loader:
            gtr_features = gtr_features.to(device)
            ney_features = ney_features.to(device)

            y_hat = model(gtr_features)
            loss = criterion(y_hat, ney_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_train_batches += 1

        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for gtr_features, ney_features, _, _ in test_data_loader:
                gtr_features = gtr_features.to(device)
                ney_features = ney_features.to(device)

                y_hat = model(gtr_features)
                loss = criterion(y_hat, ney_features)

                val_loss += loss.item()
                num_val_batches += 1

        train_loss = running_loss / num_train_batches
        val_loss = val_loss / num_val_batches

        print(
            f"E: {epoch + 1:03d}/{num_epochs}\t T: {train_loss:.6f}\t V: {val_loss:.6f}")

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if early_stop.early_stop(val_loss):
            print("Stopping early...")
            break

    return model, history


if __name__ == "__main__":
    import pickle
    import torch.optim as optim
    from models.model_0 import Model_0
    from early_stopper import EarlyStopper
    from magnitude_loss import MagnitudeLoss
    from dataset import build_data_loaders

    with open("dataset/features/min_max.pkl", "rb") as handle:
        min_max = pickle.load(handle)

    device = torch.device("cpu")
    model = Model_0()
    train_data_loader, test_data_loader = build_data_loaders(
        min_max, part="magnitude", test_size=0.1)

    criterion = MagnitudeLoss(
        min_max["ney"]["min"]["magnitude"],
        min_max["ney"]["max"]["magnitude"]
    )

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    es = EarlyStopper(5, 3e-6)
    model, history = train_model(
        model,
        criterion,
        optimizer,
        device,
        train_data_loader, test_data_loader,
        es,
        num_epochs=1
    )
