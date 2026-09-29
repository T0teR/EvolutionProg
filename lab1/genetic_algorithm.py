import numpy as np


class GeneticAlgorithm:
    def __init__(
        self,
        objective_function,
        dimension,
        lower_bound,
        upper_bound,
        population_size=50,
        crossover_probability=0.8,
        mutation_probability=0.1,
        mutation_sigma=0.5,
        tournament_size=3,
        elite_size=1,
        max_evaluations=10000,
        seed=None
    ):
        self.objective_function = objective_function

        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.population_size = population_size

        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.mutation_sigma = mutation_sigma

        self.tournament_size = tournament_size
        self.elite_size = elite_size

        self.max_evaluations = max_evaluations

        self.rng = np.random.default_rng(seed)

        self.evaluations = 0


    def initialize_population(self):
        return self.rng.uniform(
            self.lower_bound,
            self.upper_bound,
            size=(self.population_size, self.dimension)
        )

    def evaluate_population(self, population):
        fitness = []

        for individual in population:
            value = self.objective_function(individual)

            fitness.append(value)
            self.evaluations += 1

        return np.array(fitness)

    def tournament_selection(self, population, fitness):
        indices = self.rng.choice(
            len(population),
            size=self.tournament_size,
            replace=False
        )

        best_index = indices[np.argmin(fitness[indices])]

        return population[best_index].copy()

    def crossover(self, parent1, parent2):

        if self.rng.random() > self.crossover_probability:
            return parent1.copy(), parent2.copy()

        alpha = self.rng.random()

        child1 = (
            alpha * parent1
            + (1 - alpha) * parent2
        )

        child2 = (
            (1 - alpha) * parent1
            + alpha * parent2
        )

        return child1, child2


    def mutate(self, individual):

        mutated = individual.copy()

        for i in range(self.dimension):

            if self.rng.random() < self.mutation_probability:

                mutated[i] += self.rng.normal(
                    0,
                    self.mutation_sigma
                )

        mutated = np.clip(
            mutated,
            self.lower_bound,
            self.upper_bound
        )

        return mutated


    def get_elites(self, population, fitness):

        elite_indices = np.argsort(fitness)[:self.elite_size]

        return population[elite_indices].copy()

    def run(self):

        self.evaluations = 0


        population = self.initialize_population()


        fitness = self.evaluate_population(population)


        best_index = np.argmin(fitness)

        best_solution = population[best_index].copy()
        best_value = fitness[best_index]

        history = [
            (self.evaluations, best_value)
        ]

        while self.evaluations < self.max_evaluations:

            elites = self.get_elites(
                population,
                fitness
            )

            new_population = []

            for elite in elites:
                new_population.append(elite)

            while len(new_population) < self.population_size:

                parent1 = self.tournament_selection(
                    population,
                    fitness
                )

                parent2 = self.tournament_selection(
                    population,
                    fitness
                )

                child1, child2 = self.crossover(
                    parent1,
                    parent2
                )

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)

                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population)

            remaining = (
                self.max_evaluations
                - self.evaluations
            )

            population_to_evaluate = population[:remaining]

            if len(population_to_evaluate) == 0:
                break

            fitness_new = self.evaluate_population(
                population_to_evaluate
            )

            current_best_index = np.argmin(fitness_new)
            current_best_value = fitness_new[
                current_best_index
            ]

            if current_best_value < best_value:

                best_value = current_best_value

                best_solution = population_to_evaluate[
                    current_best_index
                ].copy()

            history.append(
                (self.evaluations, best_value)
            )

            if self.evaluations >= self.max_evaluations:
                break

            if len(population_to_evaluate) < self.population_size:
                break

            population = population_to_evaluate
            fitness = fitness_new

        return best_solution, best_value, history