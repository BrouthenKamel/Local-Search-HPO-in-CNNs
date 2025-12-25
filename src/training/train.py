import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
import os

from tqdm import tqdm

from src.loading.data.loader import load_dataset
from src.loading.models.model_builder import create_model
from src.schema.training import OptimizerType, TrainingParams, Epoch, History, TrainingResult
from src.schema.dataset import Dataset

# from src.loading.models.alexnet import AlexNetArchitecture
# from src.loading.models.model_builder import load_model_architecture

def count_parameters(model: torch.nn.Module) -> int:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6

def get_optimizer(model: nn.Module, training_params: TrainingParams):
    
    kwargs = {}
    if training_params.learning_rate is not None:
        kwargs['lr'] = training_params.learning_rate
    if training_params.momentum is not None:
        kwargs['momentum'] = training_params.momentum
    if training_params.weight_decay is not None:
        kwargs['weight_decay'] = training_params.weight_decay
        
    if training_params.optimizer == OptimizerType.SGD:
        return optim.SGD(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.ADAM:
        return optim.Adam(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.ADAMW:
        return optim.AdamW(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.RMSPROP:
        return optim.RMSprop(model.parameters(), **kwargs)
    elif training_params.optimizer == OptimizerType.ADGARAD:
        return optim.Adagrad(model.parameters(), **kwargs)

def evaluate_model(model: nn.Module, data_loader, criterion, device):
    """
    Evaluates the model on the provided data loader with a tqdm progress bar.

    Args:
        model (nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset to evaluate on.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to perform computation on.

    Returns:
        Tuple[float, float]: Average loss and accuracy in percentage.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            avg_loss = total_loss / total if total > 0 else 0
            accuracy = 100 * correct / total if total > 0 else 0

            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model: nn.Module, dataset, training_params: TrainingParams, patience: int = 5, optimizer: torch.optim.Optimizer = None) -> TrainingResult:
    """
    Trains the model on the provided dataset using the specified training parameters.

    Args:
        model (nn.Module): The model to train.
        dataset: An object with 'train_dataset' and 'test_dataset' attributes.
        training_params: An object with 'batch_size', 'epochs', and other training parameters.
    """
    print(f"Model parameters: {count_parameters(model):.3f} Million")
    torch.autograd.set_detect_anomaly(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = get_optimizer(model, training_params)

    train_loader = DataLoader(dataset.train_dataset, batch_size=training_params.batch_size, shuffle=True)
    test_loader = DataLoader(dataset.test_dataset, batch_size=training_params.batch_size, shuffle=False)

    history = History(epochs=[])

    best_accuracy = 0.0
    best_model_state = None
    best_optimizer_state = None
    epochs_since_improvement = 0

    for epoch in range(training_params.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params.epochs}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{train_loss / train_total:.4f}',
                'acc': f'{100 * train_correct / train_total:.2f}%'
            })

        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%")

        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} | Test Loss:  {test_loss:.4f} | Test Accuracy:  {test_accuracy:.2f}%")

        history.record_epoch(
            epoch=epoch+1,
            train_loss=avg_train_loss,
            test_loss=test_loss,
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"No improvement in {patience} epochs. Early stopping.")
                break

        print('-' * 20)

    torch.cuda.empty_cache()
    print(f"Finished Training. Best Test Accuracy: {best_accuracy:.2f}%")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)

    return TrainingResult(model, history, best_accuracy, optimizer)
