import time
import json
import os
import torch

from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName
from src.training.train import train_model
from src.schema.training import TrainingParams, OptimizerType
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small

base_dir = './records/'
os.makedirs(base_dir, exist_ok=True)

save_path = os.path.join(base_dir, 'pretraining_freeze.json')
    
record = []

training_params = TrainingParams(
    epochs=10,
    batch_size=64,
    learning_rate=1e-4,
    optimizer=OptimizerType.ADAM,
    momentum=None,
    weight_decay=1e-4,
)

freeze = [5, 6, 7]

dataset_name = DatasetName.CIFAR10
dataset = load_dataset(dataset_name, augment=True, augmentation_type='basic')
hp_space = MobileNetHPSpace(num_blocks=11)

def save_progress(record, path):
    with open(path, 'w') as f:
        json.dump(record, f, indent=4)
        
def save_model(model, path):
    torch.save(model.state_dict(), path)

for k in freeze:
    print(f"Freezeing {k} blocks...")
    
    for freeze_weights in [False]:
        print(f"Freezing weights: {freeze_weights}")

        model = MobileNetV3Small(num_classes=dataset.num_classes, pretrained=True, freeze_blocks_until=k, freeze=freeze_weights, initialize=False)

        s = time.time()
        res = train_model(model, dataset, training_params)
        e = time.time()

        training_time = round((e - s) / 60, 2)
        print(f"Training time: {training_time:.2f} minutes")

        record.append({
            'freeze_blocks': k,
            'freeze_weights': freeze_weights,
            'history': res.history.model_dump(),
            'time': training_time
        })

        save_progress(record, save_path)
        # save_model(model, os.path.join(base_dir, f'model_pretrained_{k}_blocks_{freeze_weights}_no_augment.pth'))

        print("=" * 20, '\n')

print("Training completed!")