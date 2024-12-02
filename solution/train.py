from typing import Dict, List, Optional

import numpy as np
import torch
import copy
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
) -> Dict[str, List[float]]:
    """Performs training of a provided model and provided dataset.

    Args:
        train_loader (DataLoader): DataLoader for training set.
        model (nn.Module): Model to train.
        criterion (nn.Module): Callable instance of loss function, that can be used to calculate loss for each batch.
        optimizer (optim.Optimizer): Optimizer used for updating parameters of the model.
        val_loader (Optional[DataLoader], optional): DataLoader for validation set.
            If defined, if should be used to calculate loss on validation set, after each epoch.
            Defaults to None.
        epochs (int, optional): Number of epochs (passes through dataset/dataloader) to train for.
            Defaults to 100.

    Returns:
        Dict[str, List[float]]: Dictionary with history of training.
    """
    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train"].append(avg_train_loss)

        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    y_pred = model(x_batch)
                    loss = criterion(y_pred, y_batch)
                    val_running_loss += loss.item()

            avg_val_loss = val_running_loss / len(val_loader)
            history["val"].append(avg_val_loss)

            # save the best model if the validation loss improves
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
        else:
            history["val"] = []  # If no val_loader, keep the val list empty

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}" if val_loader else f"Train Loss: {avg_train_loss:.4f}"
        )

    if val_loader is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history