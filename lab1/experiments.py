import numpy as np
import pandas as pd

from genetic_algorithm import GeneticAlgorithm
from random_search import random_search


def run_ga_experiments(
    objective_function,
    config,
    n_runs=20,
    base_seed=42
):
    results = []
    histories = []

    for run in range(n_runs):

        seed = base_seed + run

        ga = GeneticAlgorithm(
            objective_function=objective_function,
            dimension=config["dimension"],
            lower_bound=config["lower_bound"],
            upper_bound=config["upper_bound"],
            population_size=config["population_size"],
            crossover_probability=config["crossover_probability"],
            mutation_probability=config["mutation_probability"],
            mutation_sigma=config["mutation_sigma"],
            tournament_size=config["tournament_size"],
            elite_size=config["elite_size"],
            max_evaluations=config["max_evaluations"],
            seed=seed
        )

        solution, value, history = ga.run()

        results.append({
            "run": run + 1,
            "seed": seed,
            "best_value": value
        })

        histories.append(history)

        print(
            f"Запуск {run + 1:2d}/{n_runs}: "
            f"best = {value:.8f}"
        )

    return pd.DataFrame(results), histories


def calculate_statistics(results):

    values = results["best_value"].to_numpy()

    return {
        "best": np.min(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "std": np.std(values),
        "worst": np.max(values)
    }


def run_random_search_experiments(
    objective_function,
    dimension,
    lower_bound,
    upper_bound,
    max_evaluations,
    n_runs=20,
    base_seed=42
):

    results = []
    histories = []

    for run in range(n_runs):

        seed = base_seed + run

        solution, value, history = random_search(
            objective_function,
            dimension,
            lower_bound,
            upper_bound,
            max_evaluations,
            seed
        )

        results.append({
            "run": run + 1,
            "seed": seed,
            "best_value": value
        })

        histories.append(history)

        print(
            f"Random Search {run + 1:2d}/{n_runs}: "
            f"best = {value:.8f}"
        )

    return pd.DataFrame(results), histories


def histories_to_array(
    histories,
    max_evaluations,
    points=100
):

    evaluation_grid = np.linspace(
        0,
        max_evaluations,
        points,
        dtype=int
    )

    all_histories = []

    for history in histories:

        evaluations = np.array(
            [item[0] for item in history]
        )

        values = np.array(
            [item[1] for item in history]
        )

        interpolated = np.empty(
            len(evaluation_grid)
        )

        for i, evaluation in enumerate(
            evaluation_grid
        ):

            valid = np.where(
                evaluations <= evaluation
            )[0]

            if len(valid) == 0:
                interpolated[i] = values[0]
            else:
                interpolated[i] = values[valid[-1]]

        all_histories.append(interpolated)

    return evaluation_grid, np.array(all_histories)