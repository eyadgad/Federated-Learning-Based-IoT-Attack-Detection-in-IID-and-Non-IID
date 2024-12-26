import torch
import torch.nn as nn
from torch.nn import Module
from copy import deepcopy
from itertools import chain
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------
# Utility Functions for Model Operations and Loss Calculations
# -----------------------------------------------------------

def set_to_zero_model_weights(model):
    """
    Sets all model weights to zero.
    
    Args:
        model (torch.nn.Module): The model whose weights need to be set to zero.
    """
    for layer_weights in model.parameters():
        layer_weights.data.sub_(layer_weights.data)


def loss_classifier(predictions, labels):
    """
    Calculates the classification loss using CrossEntropyLoss.
    
    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground truth labels.
    
    Returns:
        torch.Tensor: Computed loss.
    """
    loss_function = nn.CrossEntropyLoss()
    return loss_function(predictions, labels.view(-1))


def loss_dataset(model, dataset, loss_f, acc=0):
    """
    Computes the total loss (and optionally accuracy) for a given model on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (torch.utils.data.DataLoader): Dataset loader to iterate through.
        loss_f (callable): Loss function to calculate the loss.
        acc (int, optional): If set to 1, returns both loss and accuracy. Defaults to 0.
    
    Returns:
        float: Loss (and optionally accuracy and correct predictions count).
    """
    loss = 0
    idx = 0
    correct = 0

    # Clone the model for validation
    model_val = deepcopy(model).to(device)
    model_val.eval()

    # Iterate over the dataset
    for idx, (features, labels) in enumerate(dataset):
        features, labels = features.to(device), labels.to(device)
        predictions = model_val(features)

        # Calculate batch loss
        batch_loss = loss_f(predictions, labels)
        loss += batch_loss.item()

        # Calculate accuracy
        _, predicted = predictions.max(1, keepdim=True)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()

    # Average loss and calculate accuracy
    loss /= (idx + 1)
    accuracy = 100 * correct / len(dataset.dataset)

    if acc == 1:
        return loss, accuracy, correct
    else:
        return loss


def difference_models_norm_2(model, model_0):
    """
    Calculates the L2 norm of the difference between two models' weights.
    
    Args:
        model (torch.nn.Module): First model.
        model_0 (torch.nn.Module): Second model to compare against.
    
    Returns:
        float: L2 norm of the difference between the two models.
    """
    tensor_1 = list(model.parameters())
    tensor_2 = list(model_0.parameters())
    sub_norm = []

    # Compute L2 norm for each layer
    for i in range(len(tensor_1)):
        s = torch.sum((tensor_1[i].to(device) - tensor_2[i].to(device)) ** 2)
        sub_norm.append(s)

    return sum(sub_norm)

# -----------------------------------------------------------
# Utility Functions for Model Operations and Loss Calculations
# -----------------------------------------------------------

def set_to_zero_model_weights(model: Module):
    """
    Set all model weights to zero.

    Args:
        model (torch.nn.Module): The model whose weights need to be zeroed.
    """
    for layer_weights in model.parameters():
        layer_weights.data.zero_()


def loss_classifier(predictions, labels):
    """
    Calculates the classification loss using CrossEntropyLoss.
    
    Args:
        predictions (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground truth labels.
    
    Returns:
        torch.Tensor: Computed loss.
    """
    loss_function = nn.CrossEntropyLoss()
    return loss_function(predictions, labels.view(-1))


def loss_dataset(model, dataset, loss_f, acc=0):
    """
    Computes the total loss (and optionally accuracy) for a given model on a dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataset (torch.utils.data.DataLoader): Dataset loader to iterate through.
        loss_f (callable): Loss function to calculate the loss.
        acc (int, optional): If set to 1, returns both loss and accuracy. Defaults to 0.
    
    Returns:
        float: Loss (and optionally accuracy and correct predictions count).
    """
    loss = 0
    idx = 0
    correct = 0

    # Clone the model for validation
    model_val = deepcopy(model).to(device)
    model_val.eval()

    # Iterate over the dataset
    for idx, (features, labels) in enumerate(dataset):
        features, labels = features.to(device), labels.to(device)
        predictions = model_val(features)

        # Calculate batch loss
        batch_loss = loss_f(predictions, labels)
        loss += batch_loss.item()

        # Calculate accuracy
        _, predicted = predictions.max(1, keepdim=True)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()

    # Average loss and calculate accuracy
    loss /= (idx + 1)
    accuracy = 100 * correct / len(dataset.dataset)

    if acc == 1:
        return loss, accuracy, correct
    else:
        return loss


def difference_models_norm_2(model, model_0):
    """
    Calculates the L2 norm of the difference between two models' weights.
    
    Args:
        model (torch.nn.Module): First model.
        model_0 (torch.nn.Module): Second model to compare against.
    
    Returns:
        float: L2 norm of the difference between the two models.
    """
    tensor_1 = list(model.parameters())
    tensor_2 = list(model_0.parameters())
    sub_norm = []

    # Compute L2 norm for each layer
    for i in range(len(tensor_1)):
        s = torch.sum((tensor_1[i].to(device) - tensor_2[i].to(device)) ** 2)
        sub_norm.append(s)

    return sum(sub_norm)

def test_step(model, test_data, loss_f):
    """
    Evaluates the model on the test dataset and computes loss and metrics.

    Args:
        model (torch.nn.Module): Model to evaluate.
        test_data (DataLoader): Test dataset.
        loss_f: Loss function.

    Returns:
        tuple: Test loss, accuracy, MAE, RMSE, and correct predictions count.
    """
    model.eval()
    test_loss, correct = 0, 0
    pred, y = [], []

    for features, labels in test_data:
        with torch.no_grad():
            features, labels = features.to(device), labels.to(device)
            predictions = model(features)
            pred.extend(list(chain.from_iterable(predictions.data.tolist())))
            y.extend(list(chain.from_iterable(labels.data.tolist())))
            batch_loss = loss_f(predictions, labels)
            test_loss += batch_loss.item()
            _, predicted = predictions.max(1, keepdim=True)
            correct += predicted.eq(labels.view_as(predicted)).sum().item()

    test_loss /= len(test_data)
    accuracy = 100 * correct / len(test_data.dataset)
    pred, y = np.array(pred), np.array(y)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))
    return test_loss, accuracy, mae, rmse, correct

def train_step(model, model_0, mu, optimizer, train_data, valid_data, loss_f, SCAFFOLD):
    """
    Executes a single training step.

    Args:
        model (torch.nn.Module): Current model.
        model_0 (torch.nn.Module): Global model for FedProx or SCAFFOLD.
        mu (float): Regularization parameter for FedProx.
        optimizer: Optimizer.
        train_data (DataLoader): Training dataset.
        valid_data (DataLoader): Validation dataset.
        loss_f: Loss function.
        SCAFFOLD (bool): If True, applies SCAFFOLD aggregation.

    Returns:
        list: Training and validation performance metrics.
    """
    tr_loss, tr_correct = 0, 0
    model.train()

    for tr_features, tr_labels in train_data:
        optimizer.zero_grad()
        tr_features, tr_labels = tr_features.to(device), tr_labels.to(device)
        tr_predictions = model(tr_features)
        tr_batch_loss = loss_f(tr_predictions, tr_labels)
        tr_prox_term = mu / 2 * difference_models_norm_2(model, model_0)
        loss = tr_batch_loss + tr_prox_term
        tr_loss += loss.item()

        loss.backward()
        optimizer.step() if not SCAFFOLD else optimizer.step(model_0.control, model.control)

        _, tr_predicted = tr_predictions.max(1, keepdim=True)
        tr_correct += tr_predicted.eq(tr_labels.view_as(tr_predicted)).sum().item()

    tr_loss /= len(train_data)
    tr_accuracy = 100 * tr_correct / len(train_data.dataset)
    val_loss, val_accuracy, val_mae, val_rmse, correct = test_step(deepcopy(model).to(device), valid_data, loss_f)
    performance = [tr_loss, tr_accuracy, val_loss, val_accuracy, val_mae, val_rmse]
    return performance

def local_learning(model, model_0, mu, train_data, test_data, epochs, loss_f, optimizer, SCAFFOLD):
    """
    Executes local training for federated learning.

    Args:
        model (torch.nn.Module): Local model.
        model_0 (torch.nn.Module): Global model.
        mu (float): Regularization parameter for FedProx.
        train_data (DataLoader): Training dataset.
        test_data (DataLoader): Test dataset.
        epochs (int): Number of training epochs.
        loss_f: Loss function.
        SCAFFOLD (bool): If True, applies SCAFFOLD updates.

    Returns:
        torch.nn.Module: Trained local model.
    """
    model.len = len(train_data)
    best_loss = np.inf
    lr_step = StepLR(optimizer, step_size=10, gamma=0.1)

    best_model = deepcopy(model).to(device)
    for epoch in range(epochs):
        performance = train_step(model, model_0, mu, optimizer, train_data, test_data, loss_f, SCAFFOLD)

        # Update model history
        model.train_loss_hist.append(performance[0])
        model.train_acc_hist.append(performance[1])
        model.val_loss_hist.append(performance[2])
        model.val_acc_hist.append(performance[3])
        model.mae_hist.append(performance[4])
        model.rmse_hist.append(performance[5])

        # Save best model
        if performance[2] < best_loss:
            best_loss = performance[2]
            best_model = deepcopy(model).to(device)
            best_model.accuracy = performance[3]
            best_model.loss = performance[2]

        print(f'Epoch {epoch + 1}/{epochs}\t Tr_Loss: {performance[0]:.4f}\t '
              f'Tr_Acc: {performance[1]:.2f}%\t Val_Loss: {performance[2]:.4f}\t '
              f'Val_Acc: {performance[3]:.2f}%\t Val_MAE: {performance[4]:.4f}\t '
              f'Val_RMSE: {performance[5]:.4f}')

        lr_step.step()

        if SCAFFOLD:
            temp = {name: param.data.clone() for name, param in model.named_parameters()}
            for name, param in model_0.named_parameters():
                local_steps = epochs * len(train_data)
                model.control[name] -= model_0.control[name]
                model.control[name] += (param.data - temp[name]) / (local_steps * model.lr)
                model.delta_y[name] = temp[name] - param.data
                model.delta_control[name] = model.control[name] - model_0.control[name]

    return deepcopy(best_model).to(device)

