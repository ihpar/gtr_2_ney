import torch


def save_model(model, model_name):
    model_scripted = torch.jit.script(model)
    model_scripted.save(f"trained_models/{model_name}.pt")


def load_model(model_name):
    model = torch.jit.load(f"trained_models/{model_name}.pt")
    model.eval()
    return model
