import numpy as np
import pytorch_lightning as pl
import sklearn.cluster
import os

def calculate_embeddings(model, dls, emb_size):
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False
    )
    batch_results = []
    
    for dl in dls:
        batch_results += trainer.predict(model, dl)

    c_pred = np.concatenate(
        list(map(lambda x: x[0].detach().cpu().numpy(), batch_results)),
        axis=0,
    )

    c_embs = np.concatenate(
        list(map(lambda x: x[1].detach().cpu().numpy(), batch_results)),
        axis=0,
    )
    c_embs = np.reshape(c_embs, (c_embs.shape[0], -1, emb_size))
    
    y_pred = np.concatenate(
        list(map(lambda x: x[2].detach().cpu().numpy(), batch_results)),
        axis=0,
    )

    return c_pred, c_embs, y_pred

def cluster_embeddings(model, emb_size, concept_embeddings_to_use, dls, save_path=None, load=True, save=True):
    if save_path is not None and load and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            return np.load(f)

    _, c_embs, _ = calculate_embeddings(model, dls, emb_size)
    c_embs = c_embs[:, concept_embeddings_to_use].reshape((c_embs.shape[0], -1))
    clusters = sklearn.cluster.HDBSCAN().fit_predict(c_embs)

    if save_path is not None and save:
        with open(save_path, "wb") as f:
            np.save(f, clusters)

    return clusters
