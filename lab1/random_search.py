import numpy as np


def random_search(
    objective_function,
    dimension,
    lower_bound,
    upper_bound,
    max_evaluations=10000,
    seed=None
):

    rng = np.random.default_rng(seed)

    best_solution = None
    best_value = float("inf")

    history = []

    for evaluation in range(1, max_evaluations + 1):

        solution = rng.uniform(
            lower_bound,
            upper_bound,
            size=dimension
        )

        value = objective_function(solution)

        if value < best_value:
            best_value = value
            best_solution = solution.copy()

        history.append(
            (evaluation, best_value)
        )

    return best_solution, best_value, history