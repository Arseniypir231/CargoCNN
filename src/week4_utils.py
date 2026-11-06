"""Pure-Python helpers for deterministic week-4 configuration and stopping."""
from __future__ import annotations

import math


class EarlyStopping:
    """Stop after `patience` epochs without a min_delta validation-loss gain."""

    def __init__(self, patience: int, min_delta: float):
        if patience < 1:
            raise ValueError("patience должен быть не меньше 1")
        if min_delta < 0:
            raise ValueError("min_delta не может быть отрицательным")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_for_patience = math.inf
        self.wait = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Return (significant_improvement, should_stop)."""
        if value < self.best_for_patience - self.min_delta:
            self.best_for_patience = value
            self.wait = 0
            return True, False
        self.wait += 1
        return False, self.wait >= self.patience


def build_experiment_configs(learning_rates=(0.001, 0.0005), batch_sizes=(8, 16),
                             dropouts=(0.0, 0.3),
                             filter_sets=((32, 64, 128, 256), (16, 32, 64, 128))) -> list[dict]:
    """Create a baseline and one-factor-at-a-time alternatives."""
    learning_rates = tuple(float(x) for x in learning_rates)
    batch_sizes = tuple(int(x) for x in batch_sizes)
    dropouts = tuple(float(x) for x in dropouts)
    filter_sets = tuple(tuple(int(v) for v in item) for item in filter_sets)
    if not learning_rates or any(x <= 0 for x in learning_rates):
        raise ValueError("Нужен список положительных Learning Rate")
    if not batch_sizes or any(x < 1 for x in batch_sizes):
        raise ValueError("Нужен список положительных Batch Size")
    if not dropouts or any(not 0 <= x < 1 for x in dropouts):
        raise ValueError("Dropout должен быть в диапазоне [0, 1)")
    if not filter_sets:
        raise ValueError("Нужен хотя бы один набор фильтров")
    for filters in filter_sets:
        if len(filters) != 4 or any(x < 1 for x in filters):
            raise ValueError("Каждый профиль filters должен содержать четыре положительных числа")

    base = {"name": "baseline", "learning_rate": learning_rates[0], "batch_size": batch_sizes[0],
            "dropout": dropouts[0], "filters": list(filter_sets[0])}
    configs = [base]
    for value in learning_rates[1:]:
        configs.append({**base, "name": f"lr_{value:g}", "learning_rate": value})
    for value in batch_sizes[1:]:
        configs.append({**base, "name": f"batch_{value}", "batch_size": value})
    for value in dropouts[1:]:
        configs.append({**base, "name": f"dropout_{value:g}", "dropout": value})
    for values in filter_sets[1:]:
        configs.append({**base, "name": "filters_" + "-".join(map(str, values)), "filters": list(values)})
    unique, seen = [], set()
    for config in configs:
        key = (config["learning_rate"], config["batch_size"], config["dropout"], tuple(config["filters"]))
        if key not in seen:
            unique.append(config)
            seen.add(key)
    return unique
