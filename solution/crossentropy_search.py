if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from loss import CrossEntropyLossLayer
    from optimizer import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer, SoftmaxLayer
    from .optimizer import SGDOptimizer
    from .loss import CrossEntropyLossLayer
    from .train import plot_model_guesses, train

from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()

def crossentropy_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the CrossEntropy problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers

    Args:
        dataset_train (TensorDataset): Dataset for training.
        dataset_val (TensorDataset): Dataset for validation.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
    """
    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=16)
    criterion = CrossEntropyLossLayer()
    learning_rate = 0.01 # default
    epochs = 50 # default
    results = {}

    # Model 1: Linear Regression Model
    model_1 = nn.Sequential(
        LinearLayer(2, 2),
        SoftmaxLayer()
    )
    optimizer_1 = SGDOptimizer(model_1.parameters(), lr=learning_rate)
    history_1 = train(train_loader, model_1, criterion, optimizer_1, val_loader, epochs)
    results["Linear Regression"] = {
        "train": history_1["train"],
        "val": history_1["val"],
        "model": model_1,
    }

    # Model 2: Single Hidden Layer with Sigmoid
    model_2 = nn.Sequential(
        LinearLayer(2, 2),
        SigmoidLayer(),
        LinearLayer(2, 2),
        SoftmaxLayer()
    )
    optimizer_2 = SGDOptimizer(model_2.parameters(), lr=learning_rate)
    history_2 = train(train_loader, model_2, criterion, optimizer_2, val_loader, epochs)
    results["1 Hidden Layer (Sigmoid)"] = {
        "train": history_2["train"],
        "val": history_2["val"],
        "model": model_2,
    }

    # Model 3: Single Hidden Layer with ReLU
    model_3 = nn.Sequential(
        LinearLayer(2, 2),
        ReLULayer(),
        LinearLayer(2, 2),
        SoftmaxLayer()
    )
    optimizer_3 = SGDOptimizer(model_3.parameters(), lr=learning_rate)
    history_3 = train(train_loader, model_3, criterion, optimizer_3, val_loader, epochs)
    results["1 Hidden Layer (ReLU)"] = {
        "train": history_3["train"],
        "val": history_3["val"],
        "model": model_3,
    }

    # Model 4: Two Hidden Layers (Sigmoid, ReLU)
    model_4 = nn.Sequential(
        LinearLayer(2, 2),
        SigmoidLayer(),
        LinearLayer(2, 2),
        ReLULayer(),
        LinearLayer(2, 2),
        SoftmaxLayer()
    )
    optimizer_4 = SGDOptimizer(model_4.parameters(), lr=learning_rate)
    history_4 = train(train_loader, model_4, criterion, optimizer_4, val_loader, epochs)
    results["2 Hidden Layers (Sigmoid, ReLU)"] = {
        "train": history_4["train"],
        "val": history_4["val"],
        "model": model_4,
    }

    # Model 5: Two Hidden Layers (ReLU, Sigmoid)
    model_5 = nn.Sequential(
        LinearLayer(2, 2),
        ReLULayer(),
        LinearLayer(2, 2),
        SigmoidLayer(),
        LinearLayer(2, 2),
        SoftmaxLayer()
    )
    optimizer_5 = SGDOptimizer(model_5.parameters(), lr=learning_rate)
    history_5 = train(train_loader, model_5, criterion, optimizer_5, val_loader, epochs)
    results["2 Hidden Layers (ReLU, Sigmoid)"] = {
        "train": history_5["train"],
        "val": history_5["val"],
        "model": model_5,
    }

    return results


def accuracy_score(model, dataloader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for CrossEntropy.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is an integer representing a correct class to a corresponding observation.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].
    """
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for observations, targets in dataloader:
            outputs = model(observations)
            predicted_classes = torch.argmax(outputs, dim=1)
            true_classes = targets

            correct += (predicted_classes == true_classes).sum().item()
            total += targets.size(0)

    return correct / total

def main():
    """
    Main function of the Crossentropy problem.
    It should:
        1. Call crossentropy_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 10 lines total).
            x-axis should be epochs, y-axis should me Crossentropy loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y))
    dataset_val = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val))
    dataset_test = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test))

    ce_configs = crossentropy_parameter_search(dataset_train, dataset_val)

    plt.figure(figsize=(10, 6))
    for model_name, data in ce_configs.items():
        plt.plot(data["train"], label=f"{model_name} - Train Loss")
        plt.plot(data["val"], label=f"{model_name} - Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("CrossEntropy Loss")
    plt.title("Training and Validation Losses (CrossEntropy)")
    plt.legend()
    plt.grid()
    plt.show()

    best_model_name = min(ce_configs, key=lambda name: min(ce_configs[name]["val"]))
    best_model_data = ce_configs[best_model_name]
    best_model = best_model_data["model"]

    print(f"Best model: {best_model_name}")
    print(f"Lowest validation loss: {min(best_model_data['val']):.4f}")

    test_loader = DataLoader(dataset_test, batch_size=16)
    plot_model_guesses(test_loader, best_model, title=f"Best Model Predictions ({best_model_name})")

    accuracy = accuracy_score(best_model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
