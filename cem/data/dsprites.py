import numpy as np
import torch

def quadrant_shape(latents):
    loc = latents[:, (False, False, False, False, True, True)] > 15
    quadrant = loc[:, 0] * 2 + loc[:, 1]
    shape = latents[:, (False, True, False, False, False, False)].squeeze()
    c = torch.stack((
        quadrant == 0,
        quadrant == 1,
        quadrant == 2,
        quadrant == 3,
        shape == 0,
        shape == 1,
        shape == 2), dim=1).float()
    y = shape*4 + quadrant

    return c, y

def quadrant_shape_shape_hidden(latents):
    loc = latents[:, (False, False, False, False, True, True)] > 15
    quadrant = loc[:, 0] * 2 + loc[:, 1]
    shape = latents[:, (False, True, False, False, False, False)].squeeze()
    c = torch.stack((
        quadrant == 0,
        quadrant == 1,
        quadrant == 2,
        quadrant == 3), dim=1).float()
    y = shape*4 + quadrant

    return c, y

loaded_datasets = {}
def load_dsprites(name, source, concepts_and_label, filter=None, permutation=None):
    dataset_zip = np.load(source)

    if permutation is None:
        permutation = np.random.permutation(len(dataset_zip["imgs"]))

    if filter is None:
        filter = lambda x: np.repeat(True, x.shape[0])

    permutation = permutation[filter(dataset_zip["latents_classes"][permutation])]

    x = torch.tensor(dataset_zip["imgs"][permutation])[:, None, :] / 255.0

    c, y = concepts_and_label(torch.tensor(dataset_zip["latents_classes"])[permutation])


    l = len(x)
    x_train, c_train, y_train = x[:int(0.6*l)], c[:int(0.6*l)], y[:int(0.6*l)]
    x_val, c_val, y_val = x[int(0.6*l):int(0.75*l)], c[int(0.6*l):int(0.75*l)], y[int(0.6*l):int(0.75*l)]
    x_test, c_test, y_test = x[int(0.75*l):], c[int(0.75*l):], y[int(0.75*l):]
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train, c_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val, c_val)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test, c_test)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=256)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=256)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=256)

    loaded_datasets[name] = (train_dl, val_dl, test_dl)
    return train_dl, val_dl, test_dl

def get_dsprites(name):
    return loaded_datasets[name]
