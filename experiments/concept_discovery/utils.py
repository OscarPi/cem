import torch
import numpy as np
import os
import cem.train.training as training
import cem.data.mnist_add as mnist
import cem.data.dsprites as dsprites

def get_dsprites_c_extractor_arch():
    def c_extractor_arch(output_dim):
        output_dim = output_dim or 128
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64*64, 1200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1200, 1200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1200, output_dim),
        )
    return c_extractor_arch

def get_dsprites_reconstruction_arch():
    def reconstruction_arch(bottleneck_size):
        return torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, 1200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1200, 1200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1200, 64*64),
            torch.nn.Unflatten(1, (1, 64, 64)),
            torch.nn.Sigmoid()
        )

    return reconstruction_arch

def get_mnist_c_extractor_arch(input_shape, num_operands):
    def c_extractor_arch(output_dim):
        output_dim = output_dim or 128
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(int(np.prod(input_shape[2:]))*num_operands, 400),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(
                400,
                output_dim,
            ),
        )
    return c_extractor_arch

def get_mnist_reconstruction_arch(input_shape, num_operands):
    def reconstruction_arch(bottleneck_size):
        return torch.nn.Sequential(
            torch.nn.Linear(bottleneck_size, 400),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(400, 400),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(400, int(np.prod(input_shape[2:]))*num_operands),
            torch.nn.Unflatten(1, (num_operands, input_shape[2], input_shape[3])),
            torch.nn.Sigmoid()
        )

    return reconstruction_arch

def get_base_config():
    return {
        "num_workers": 0,
        "weight_loss": True,
        "top_k_accuracy": None,
        "emb_size": 16,
        "extra_dims": 0,
        "concept_loss_weight": 10,
        "learning_rate": 0.001,
        "weight_decay": 4e-06,
        "optimizer": "sgd",
        "bool": False,
        "sigmoidal_prob": True,
        "training_intervention_prob": 0.25,
        "intervention_freq": 1,
        "c2y_layers": [128, 128],
        "architecture": "ConceptEmbeddingModel",
        "shared_prob_gen": True,
        "embedding_activation": "leakyrelu",
        "max_epochs": 300,
        "skip_repr_evaluation": True,
        "test_subsampling": 1,
        "use_task_class_weights": True,
        "check_val_every_n_epoch": 2,
        "save_model": True,
        "patience": 300,
        "early_stopping_monitor": "val_loss",
        "early_stopping_mode": "min",
        "early_stopping_delta": 0.0,
        "momentum": 0.9,
        "extra_name": "",
        "sigmoidal_embedding": False,
        "concat_prob": False,
        "include_certainty": False,
        "imbalance": None
    }

def get_mnist_config(selected_digits, threshold_labels, sampling_percent, reconstruction_loss=False):
    batch_size = 2048
    input_shape=(
        batch_size,
        len(selected_digits),
        28,
        28,
    )
    return {
        **get_base_config(),
        "selected_digits": selected_digits,
        "num_operands": len(selected_digits),
        "threshold_labels": threshold_labels,
        "sampling_percent": sampling_percent,
        "sampling_groups": True,
        "train_dataset_size": 10000,
        "batch_size": batch_size,
        "noise_level": 0.0,
        "c_extractor_arch": get_mnist_c_extractor_arch(input_shape, len(selected_digits)),
        "reconstruction_arch": get_mnist_reconstruction_arch(input_shape, len(selected_digits)) if reconstruction_loss else None
    }

def get_dsprites_config(n_concepts, n_tasks, reconstruction_loss=False):
    return {
        **get_base_config(),
        "c_extractor_arch": get_dsprites_c_extractor_arch(),
        "reconstruction_arch": get_dsprites_reconstruction_arch() if reconstruction_loss else None,
        "n_concepts": n_concepts,
        "n_tasks": n_tasks
    }

def train_model(config, train_dl, val_dl, test_dl, save_path):
    attribute_count = np.zeros((max(config["n_tasks"], 2),))
    samples_seen = 0
    for data in train_dl:
        if len(data) == 2:
            (_, (y, _)) = data
        else:
            (_, y, _) = data
        if config["n_tasks"] > 1:
            y = torch.nn.functional.one_hot(
                y,
                num_classes=config["n_tasks"],
            ).cpu().detach().numpy()
        else:
            y = torch.cat(
                [torch.unsqueeze(1 - y, dim=-1), torch.unsqueeze(y, dim=-1)],
                dim=-1,
            ).cpu().detach().numpy()
        attribute_count += np.sum(y, axis=0)
        samples_seen += y.shape[0]
    print("Class distribution is:", attribute_count / samples_seen)
    if config["n_tasks"] > 1:
        task_class_weights = samples_seen / attribute_count - 1
    else:
        task_class_weights = np.array(
            [attribute_count[0]/attribute_count[1]]
        )

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    return training.train_model(
        task_class_weights=task_class_weights,
        accelerator="auto",
        devices="auto",
        n_concepts=config["n_concepts"],
        n_tasks=config["n_tasks"],
        config=config,
        train_dl=train_dl,
        val_dl=val_dl,
        test_dl=test_dl,
        result_dir=save_path,
        seed=42,
        imbalance=config.get("imbalance", None)
    )

def train_mnist_model(n_digits, n_concepts, save_path, sum_as_label=False, reconstruction_loss=False):
    mnist.dls(n_digits, n_concepts, sum_as_label)
    config = get_mnist_config(
        selected_digits=[[0, 1]] * n_digits,
        threshold_labels=None if sum_as_label else int(np.ceil(n_digits / 2)),
        sampling_percent=n_concepts / n_digits,
        reconstruction_loss=reconstruction_loss
    )
    config.update(mnist.loaded_dataset_metadata[(n_digits, n_concepts, sum_as_label)])
    return train_model(
        config,
        mnist.train_dl(n_digits, n_concepts, sum_as_label),
        mnist.val_dl(n_digits, n_concepts, sum_as_label),
        mnist.test_dl(n_digits, n_concepts, sum_as_label),
        save_path)

def train_dsprites_model(dsprites_name, n_concepts, n_tasks, save_path, reconstruction_loss=False):
    train_dl, val_dl, test_dl = dsprites.get_dsprites(dsprites_name)
    config = get_dsprites_config(n_concepts, n_tasks, reconstruction_loss=reconstruction_loss)
    return train_model(config, train_dl, val_dl, test_dl, save_path)
