"""
Генерация синтетических данных для задачи «Рюкзак с конфликтующими предметами».
Вариант V=2, Lab 2.
"""

import json
import random
from pathlib import Path

SEED = 42
N_ITEMS = 40
N_CONFLICTS = 18
CAPACITY_RATIO = 0.35   # доля от суммарного веса


def generate(seed: int = SEED):
    rng = random.Random(seed)

    items = []
    for i in range(N_ITEMS):
        weight = rng.randint(5, 25)
        value = rng.randint(20, 120)
        items.append({
            "id": i,
            "name": f"Item_{i:02d}",
            "weight": weight,
            "value": value
        })

    total_weight = sum(it["weight"] for it in items)
    capacity = int(total_weight * CAPACITY_RATIO)

    # Генерация конфликтующих пар (без повторов и самоконфликтов)
    possible_pairs = [(i, j) for i in range(N_ITEMS) for j in range(i + 1, N_ITEMS)]
    conflicts = rng.sample(possible_pairs, N_CONFLICTS)
    conflicts = sorted(conflicts)

    data = {
        "seed": seed,
        "n_items": N_ITEMS,
        "capacity": capacity,
        "items": items,
        "conflicts": conflicts,
        "description": (
            "0/1 Knapsack with pairwise conflicts. "
            "A solution is feasible if total weight ≤ capacity "
            "and no conflicting pair is both selected."
        )
    }
    return data


def save(data, path: str = "data/knapsack_conflicts.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Data saved to {path}")
    print(f"Items: {data['n_items']}, Capacity: {data['capacity']}, Conflicts: {len(data['conflicts'])}")


if __name__ == "__main__":
    data = generate()
    save(data)
