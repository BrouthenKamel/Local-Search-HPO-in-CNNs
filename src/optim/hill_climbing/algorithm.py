import torch
import time
import random

from src.loading.models.mobilenet.hp import MobileNetHP
from src.loading.models.mobilenet.space import MobileNetHPSpace
from src.training.train import train_model, count_parameters
from src.schema.training import TrainingParams, OptimizerType

from src.loading.models.mobilenet.config import MobileNetConfig
from src.loading.models.mobilenet.model import MobileNetV3Small

def evaluate(model, dataset, epochs: int, optimizer = None):
    training_params = TrainingParams(
        epochs=epochs,
        batch_size=64,
        learning_rate=1e-4,
        optimizer=OptimizerType.ADAM,
        momentum=None,
        weight_decay=1e-4,
    )
    
    start_time = time.time()
    train_results = train_model(model, dataset, training_params, optimizer=optimizer)
    eval_time = (time.time() - start_time) / 60

    best_test_accuracy = train_results.best_test_accuracy
    # test_accuracy = random.random()

    print(f"Evaluated model with test_accuracy={best_test_accuracy:.4f}, time={eval_time:.2f} minutes")

    return train_results.model, train_results.optimizer, best_test_accuracy
    # return model, None, test_accuracy

def hill_climbing_optimization(
    initial_hp: MobileNetHP,
    hp_space: MobileNetHPSpace,
    dataset,
    iterations: int = 10,
    neighbors_per_iteration: int = 4,
    max_epochs: int = 20,
    block_modification_ratio: float = 0.5,
    param_modification_ratio: float = 0.5,
    perturbation_intensity: int = 2,
    perturbation_strategy: str = "local",
    freeze_blocks_until: int = 0
):
    stage_schedule = [max_epochs // 5, int(max_epochs / 2.5), max_epochs]
    pretrained = freeze_blocks_until > 0

    # Initial evaluation
    current_hp = initial_hp
    config = MobileNetConfig.from_hp(current_hp)
    model = MobileNetV3Small(config, dataset.num_classes, pretrained=pretrained, freeze_blocks_until=freeze_blocks_until)
    optimizer = None

    _, optimizer, current_perf = evaluate(model, dataset, max_epochs, optimizer)
    history = [{
        'iteration': 0,
        'best_hp': current_hp.to_dict(),
        'best_perf': current_perf,
        'parameters': count_parameters(model)
    }]

    for iter_idx in range(iterations):
        print(f"\n=== Iteration {iter_idx+1}/{iterations} ===")
        stage1_epochs = stage_schedule[0]
        stage2_epochs = stage_schedule[1]
        stage3_epochs = stage_schedule[2]
        pretrained = freeze_blocks_until > 0

        # Stage 1: Train all neighbors for stage1_epochs
        print(f"\n→ Stage 1: Training {neighbors_per_iteration} candidates for {stage1_epochs} epochs")
        candidates = []
        while len(candidates) < neighbors_per_iteration:
            candidate_hp = hp_space.neighbor(
                current_hp,
                block_modification_ratio,
                param_modification_ratio,
                perturbation_intensity,
                perturbation_strategy
            )
            model = MobileNetV3Small(
                MobileNetConfig.from_hp(candidate_hp),
                dataset.num_classes,
                pretrained=pretrained,
                freeze_blocks_until=freeze_blocks_until
            )

            try:
                model, optimizer, acc = evaluate(model, dataset, stage1_epochs, optimizer=None)
                candidates.append({
                    'hp': candidate_hp,
                    'model': model,
                    'optimizer': optimizer,
                    'score': acc
                })
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"Candidate skipped during Stage 1 due to error: {e}")

        # Stage 2: Train top 50% for (stage2 - stage1) additional epochs
        additional_stage2_epochs = stage2_epochs - stage1_epochs
        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates[:max(1, len(candidates) // 2)]

        print(f"\n→ Stage 2: Training {len(candidates)} candidates for {additional_stage2_epochs} more epochs")
        for i, candidate in enumerate(candidates):
            try:
                model, optimizer, acc = evaluate(candidate['model'], dataset, additional_stage2_epochs, candidate['optimizer'])
                candidate['model'] = model
                candidate['optimizer'] = optimizer
                candidate['score'] = acc
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"Candidate {i+1} failed during Stage 2: {e}")

        candidates.sort(key=lambda x: x['score'], reverse=True)
        candidates = candidates[:1]

        # Final stage: continue best candidate for remaining epochs to reach max
        best_candidate = candidates[0]
        additional_stage3_epochs = stage3_epochs - stage2_epochs

        print(f"\n→ Final Stage: Continuing best candidate for {additional_stage3_epochs} more epochs")

        try:
            final_model, final_opt, final_perf = evaluate(
                best_candidate['model'], dataset, additional_stage3_epochs, best_candidate['optimizer']
            )
        except Exception as e:
            torch.cuda.empty_cache()
            print("Final evaluation failed:", e)
            final_perf = -1

        if final_perf > current_perf:
            print(f"New best model found! Accuracy: {final_perf:.4f}")
            current_hp = best_candidate['hp']
            current_perf = final_perf
        else:
            print("Final model did not outperform current best.")

        history.append({
            'iteration': iter_idx + 1,
            'best_hp': current_hp.to_dict(),
            'best_perf': current_perf,
            'parameters': count_parameters(final_model)
        })
        print(f"Iteration {iter_idx+1} complete. Best so far: {current_perf:.4f}")

    print("\nOptimization finished.")
    return current_hp, history

