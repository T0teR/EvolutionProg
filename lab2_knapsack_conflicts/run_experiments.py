"""
Серия экспериментов для лабораторной работы № 2 (вариант V=2).
"""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

from generate_data import generate, save
from ga_knapsack import KnapsackConflicts, GeneticAlgorithm, greedy_heuristic, random_feasible_search


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_series(
    problem: KnapsackConflicts,
    n_runs: int = 25,
    constraint_mode: str = "repair",
    crossover_type: str = "one_point",
    pop_size: int = 80,
    generations: int = 120,
    mutation_rate: float = 0.02,
    penalty_coef: float = 50.0,
    base_seed: int = 1000,
) -> List[Dict[str, Any]]:
    results = []
    for r in range(n_runs):
        seed = base_seed + r
        ga = GeneticAlgorithm(
            problem=problem,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=0.9,
            mutation_rate=mutation_rate,
            tournament_size=3,
            elitism=2,
            constraint_mode=constraint_mode,
            crossover_type=crossover_type,
            penalty_coef=penalty_coef,
            seed=seed,
        )
        t0 = time.time()
        res = ga.run()
        res["run"] = r
        res["seed"] = seed
        res["time_sec"] = time.time() - t0
        res["constraint_mode"] = constraint_mode
        res["crossover_type"] = crossover_type
        results.append(res)
        print(f"  run {r+1:02d}/{n_runs}: value={res['best_value']:.0f}, "
              f"weight={res['best_weight']}, feasible={res['is_feasible']}, "
              f"time={res['time_sec']:.2f}s")
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, float]:
    values = [r["best_value"] for r in results]
    return {
        "n_runs": len(values),
        "best": float(np.max(values)),
        "worst": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)),
        "feasible_rate": float(np.mean([r["is_feasible"] for r in results])),
    }


def save_csv(results: List[Dict[str, Any]], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "seed", "best_value", "best_weight", "is_feasible",
                        "time_sec", "constraint_mode", "crossover_type", "n_selected"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "run": r["run"],
                "seed": r["seed"],
                "best_value": r["best_value"],
                "best_weight": r["best_weight"],
                "is_feasible": r["is_feasible"],
                "time_sec": round(r["time_sec"], 3),
                "constraint_mode": r["constraint_mode"],
                "crossover_type": r["crossover_type"],
                "n_selected": len(r["selected_items"]),
            })


def plot_convergence(results: List[Dict[str, Any]], title: str, path: Path):
    """Средняя траектория + min/max по поколениям."""
    histories = [r["best_history"] for r in results]
    max_len = max(len(h) for h in histories)
    # выравниваем
    padded = []
    for h in histories:
        if len(h) < max_len:
            h = h + [h[-1]] * (max_len - len(h))
        padded.append(h)
    arr = np.array(padded)

    gens = np.arange(max_len)
    mean = arr.mean(axis=0)
    min_ = arr.min(axis=0)
    max_ = arr.max(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(gens, mean, label="Среднее лучшее", color="C0", linewidth=2)
    plt.fill_between(gens, min_, max_, alpha=0.25, color="C0", label="min–max")
    plt.xlabel("Поколение")
    plt.ylabel("Лучшее значение целевой функции")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved: {path}")


def main():
    print("=" * 60)
    print("Lab 2 | Variant V=2 | Knapsack with conflicts")
    print("=" * 60)

    # 1. Данные
    data = generate(seed=42)
    save(data, "data/knapsack_conflicts.json")
    problem = KnapsackConflicts(data)

    print(f"\nProblem: {problem.n} items, capacity={problem.capacity}, "
          f"conflicts={len(problem.conflicts)}")

    # 2. Базовые методы
    print("\n--- Greedy heuristic ---")
    greedy = greedy_heuristic(problem)
    print(f"Value: {greedy['value']}, Weight: {greedy['weight']}, "
          f"Items: {len(greedy['selected_items'])}")

    print("\n--- Random feasible search (5000 trials) ---")
    rnd = random_feasible_search(problem, n_trials=5000, seed=999)
    print(f"Value: {rnd['value']}, Weight: {rnd['weight']}, "
          f"Items: {len(rnd['selected_items'])}")

    # 3. Основные серии
    configs = [
        {"name": "repair_onepoint", "constraint_mode": "repair", "crossover_type": "one_point"},
        {"name": "repair_uniform",  "constraint_mode": "repair", "crossover_type": "uniform"},
        {"name": "penalty_onepoint","constraint_mode": "penalty","crossover_type": "one_point", "penalty_coef": 40.0},
        {"name": "penalty_strong",  "constraint_mode": "penalty","crossover_type": "one_point", "penalty_coef": 120.0},
    ]

    all_summaries = {}
    best_overall = None
    best_overall_val = -1

    for cfg in configs:
        print(f"\n=== Configuration: {cfg['name']} ===")
        results = run_series(
            problem,
            n_runs=25,
            constraint_mode=cfg["constraint_mode"],
            crossover_type=cfg["crossover_type"],
            pop_size=80,
            generations=120,
            mutation_rate=0.025,
            penalty_coef=cfg.get("penalty_coef", 50.0),
            base_seed=2000,
        )
        summary = summarize(results)
        all_summaries[cfg["name"]] = summary
        print("Summary:", summary)

        save_csv(results, RESULTS_DIR / f"{cfg['name']}_runs.csv")
        plot_convergence(
            results,
            title=f"Convergence — {cfg['name']}",
            path=RESULTS_DIR / f"{cfg['name']}_convergence.png"
        )

        # сохраняем лучший запуск
        best_run = max(results, key=lambda r: r["best_value"])
        if best_run["best_value"] > best_overall_val:
            best_overall_val = best_run["best_value"]
            best_overall = best_run
            best_overall["config"] = cfg["name"]

        # сохраняем детали лучшего
        with open(RESULTS_DIR / f"{cfg['name']}_best.json", "w", encoding="utf-8") as f:
            json.dump({
                "config": cfg,
                "value": best_run["best_value"],
                "weight": best_run["best_weight"],
                "selected_items": best_run["selected_items"],
                "seed": best_run["seed"],
            }, f, ensure_ascii=False, indent=2)

    # 4. Сводная таблица
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Config':<22} {'Best':>8} {'Mean':>8} {'Median':>8} {'Std':>8} {'Feas%':>7}")
    for name, s in all_summaries.items():
        print(f"{name:<22} {s['best']:8.1f} {s['mean']:8.1f} {s['median']:8.1f} "
              f"{s['std']:8.2f} {s['feasible_rate']*100:6.1f}%")

    print(f"\nGreedy:          {greedy['value']}")
    print(f"Random search:   {rnd['value']}")

    # 5. Лучшее найденное решение (для отчёта)
    print("\n" + "=" * 60)
    print("BEST SOLUTION OVERALL")
    print("=" * 60)
    print(f"Config: {best_overall['config']}")
    print(f"Value:  {best_overall['best_value']}")
    print(f"Weight: {best_overall['best_weight']} / {problem.capacity}")
    print(f"Items:  {best_overall['selected_items']}")
    print(f"Seed:   {best_overall['seed']}")

    # Сохраняем сводку
    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "summaries": all_summaries,
            "greedy": {"value": greedy["value"], "weight": greedy["weight"],
                       "selected": greedy["selected_items"]},
            "random_search": {"value": rnd["value"], "weight": rnd["weight"],
                              "selected": rnd["selected_items"]},
            "best_overall": {
                "config": best_overall["config"],
                "value": best_overall["best_value"],
                "weight": best_overall["best_weight"],
                "selected_items": best_overall["selected_items"],
                "seed": best_overall["seed"],
            }
        }, f, ensure_ascii=False, indent=2)

    print("\nAll results saved to ./results/")


if __name__ == "__main__":
    main()
