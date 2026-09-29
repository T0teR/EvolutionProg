import os

import numpy as np
import matplotlib.pyplot as plt

from functions import rosenbrock

from experiments import (
    run_ga_experiments,
    calculate_statistics,
    run_random_search_experiments,
    histories_to_array
)



DIMENSION = 7

LOWER_BOUND = -5.0
UPPER_BOUND = 10.0

N_RUNS = 20

MAX_EVALUATIONS = 10000

BASE_SEED = 42



CONFIG_1 = {

    "dimension": DIMENSION,

    "lower_bound": LOWER_BOUND,
    "upper_bound": UPPER_BOUND,

    "population_size": 50,

    "crossover_probability": 0.8,

    "mutation_probability": 0.1,

    "mutation_sigma": 0.5,

    "tournament_size": 3,

    "elite_size": 1,

    "max_evaluations": MAX_EVALUATIONS
}


CONFIG_2 = {

    "dimension": DIMENSION,

    "lower_bound": LOWER_BOUND,
    "upper_bound": UPPER_BOUND,

    "population_size": 100,

    "crossover_probability": 0.8,

    "mutation_probability": 0.1,

    "mutation_sigma": 0.5,

    "tournament_size": 3,

    "elite_size": 1,

    "max_evaluations": MAX_EVALUATIONS
}


os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)


print("=" * 60)
print("КОНФИГУРАЦИЯ 1")
print("=" * 60)

results_1, histories_1 = run_ga_experiments(
    rosenbrock,
    CONFIG_1,
    N_RUNS,
    BASE_SEED
)

stats_1 = calculate_statistics(results_1)

print("\nСтатистика конфигурации 1:")

for key, value in stats_1.items():
    print(f"{key:>8}: {value:.8f}")


print("\n" + "=" * 60)
print("КОНФИГУРАЦИЯ 2")
print("=" * 60)

results_2, histories_2 = run_ga_experiments(
    rosenbrock,
    CONFIG_2,
    N_RUNS,
    BASE_SEED
)

stats_2 = calculate_statistics(results_2)

print("\nСтатистика конфигурации 2:")

for key, value in stats_2.items():
    print(f"{key:>8}: {value:.8f}")



results_1.to_csv(
    "results/ga_config_1.csv",
    index=False
)

results_2.to_csv(
    "results/ga_config_2.csv",
    index=False
)



x1, history_array_1 = histories_to_array(
    histories_1,
    MAX_EVALUATIONS
)

x2, history_array_2 = histories_to_array(
    histories_2,
    MAX_EVALUATIONS
)


mean_1 = np.mean(
    history_array_1,
    axis=0
)

min_1 = np.min(
    history_array_1,
    axis=0
)

max_1 = np.max(
    history_array_1,
    axis=0
)


mean_2 = np.mean(
    history_array_2,
    axis=0
)

min_2 = np.min(
    history_array_2,
    axis=0
)

max_2 = np.max(
    history_array_2,
    axis=0
)


plt.figure(figsize=(10, 6))

plt.plot(
    x1,
    mean_1,
    label="GA, population = 50"
)

plt.fill_between(
    x1,
    min_1,
    max_1,
    alpha=0.2
)

plt.plot(
    x2,
    mean_2,
    label="GA, population = 100"
)

plt.fill_between(
    x2,
    min_2,
    max_2,
    alpha=0.2
)

plt.xlabel(
    "Количество вычислений целевой функции"
)

plt.ylabel(
    "Лучшее значение функции"
)

plt.title(
    "Сходимость генетического алгоритма"
)

plt.yscale("log")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/convergence_comparison.png",
    dpi=300
)

plt.show()


print("\n" + "=" * 60)
print("СЛУЧАЙНЫЙ ПОИСК")
print("=" * 60)

random_results, random_histories = (
    run_random_search_experiments(
        rosenbrock,
        DIMENSION,
        LOWER_BOUND,
        UPPER_BOUND,
        MAX_EVALUATIONS,
        N_RUNS,
        BASE_SEED
    )
)

random_stats = calculate_statistics(
    random_results
)

print("\nСтатистика случайного поиска:")

for key, value in random_stats.items():
    print(f"{key:>8}: {value:.8f}")


random_results.to_csv(
    "results/random_search.csv",
    index=False
)


print("\n" + "=" * 60)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("=" * 60)

print(
    f"ГА, population = 50: "
    f"mean = {stats_1['mean']:.8f}"
)

print(
    f"ГА, population = 100: "
    f"mean = {stats_2['mean']:.8f}"
)

print(
    f"Random Search: "
    f"mean = {random_stats['mean']:.8f}"
)


x_random, random_history_array = (
    histories_to_array(
        random_histories,
        MAX_EVALUATIONS
    )
)

random_mean = np.mean(
    random_history_array,
    axis=0
)

random_min = np.min(
    random_history_array,
    axis=0
)

random_max = np.max(
    random_history_array,
    axis=0
)


plt.figure(figsize=(10, 6))

plt.plot(
    x1,
    mean_1,
    label="GA, population = 50"
)

plt.plot(
    x2,
    mean_2,
    label="GA, population = 100"
)

plt.plot(
    x_random,
    random_mean,
    label="Random Search"
)

plt.xlabel(
    "Количество вычислений целевой функции"
)

plt.ylabel(
    "Лучшее значение функции"
)

plt.title(
    "Генетический алгоритм и случайный поиск"
)

plt.yscale("log")

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/ga_vs_random.png",
    dpi=300
)

plt.show()