from enum import Enum
from typing import Optional, List
from pydantic import BaseModel
import torch.nn as nn
import torch

class OptimizerType(str, Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    ADAMW = 'adamw'
    RMSPROP = 'rmsprop'
    ADAGRAD = 'adagrad'

class TrainingParams(BaseModel):
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: OptimizerType
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None

class Epoch(BaseModel):
    epoch: int
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float

class History(BaseModel):
    epochs: List[Epoch]

    def record_epoch(self, epoch: int, train_loss: float, test_loss: float, train_accuracy: float, test_accuracy: float):
        self.epochs.append(Epoch(epoch=epoch, train_loss=train_loss, test_loss=test_loss, train_accuracy=train_accuracy, test_accuracy=test_accuracy))

class TrainingResult:
    def __init__(self, model: nn.Module, history: History, best_test_accuracy: float, optimizer: torch.optim.Optimizer = None):
        self.model = model
        self.history = history
        self.best_test_accuracy = best_test_accuracy
        self.optimizer = optimizer
