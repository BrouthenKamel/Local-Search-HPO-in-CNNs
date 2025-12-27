import warnings
warnings.filterwarnings("ignore")

import json

from src.loading.models.mobilenet.hp import original_hp
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.loading.data.loader import load_dataset
from src.schema.dataset import DatasetName
from src.optim.hill_climbing.algorithm import hill_climbing_optimization

# Load dataset
dataset = load_dataset(DatasetName.CIFAR10, augment=True, augmentation_type='auto')

# Initialize hyperparameter space
f = 9
hp_space = MobileNetHPSpace(freeze_blocks_until=f)

# Run hill climbing optimization
best_hp, history = hill_climbing_optimization(
    initial_hp=original_hp,
    hp_space=hp_space,
    dataset=dataset,
    iterations=10,
    neighbors_per_iteration=4,
    max_epochs=20,
    block_modification_ratio=0.8,
    param_modification_ratio=0.5,
    perturbation_intensity=2,
    perturbation_strategy="local",
    freeze_blocks_until=f
)

# Print results
print("=" * 40)
print("Best Hyperparameters Found:")
print(best_hp.to_dict())
print("=" * 40)

name = f"it_10_n_4_ep_20_blr_0.8_pmr_0.5_pi_2_local_freeze_{f}"
record_dir = "src/optim/hill_climbing/records"

with open(f"{record_dir}/{name}.json", 'w') as f:
    json.dump(history, f, indent=4)

print(f"History saved to {record_dir}/{name}.json")