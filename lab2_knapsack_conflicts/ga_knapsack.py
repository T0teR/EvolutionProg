"""
Генетический алгоритм для задачи «Рюкзак с конфликтующими предметами».
Вариант V=2 (Lab 2).

Представление: бинарная строка длины n.
Ограничения:
  1. Суммарный вес ≤ capacity
  2. Ни одна конфликтующая пара не выбрана одновременно

Два способа работы с ограничениями:
  - repair  (основной)
  - penalty
"""

from __future__ import annotations
import random
import copy
from typing import List, Tuple, Dict, Any
import numpy as np


class KnapsackConflicts:
    def __init__(self, data: Dict[str, Any]):
        self.n = data["n_items"]
        self.capacity = data["capacity"]
        self.items = data["items"]
        self.weights = np.array([it["weight"] for it in self.items], dtype=int)
        self.values = np.array([it["value"] for it in self.items], dtype=int)
        self.conflicts = data["conflicts"]  # list of (i, j)

        # Для быстрого доступа: для каждого предмета список конфликтующих с ним
        self.conflict_sets = [set() for _ in range(self.n)]
        for i, j in self.conflicts:
            self.conflict_sets[i].add(j)
            self.conflict_sets[j].add(i)

    def is_feasible(self, chrom: np.ndarray) -> bool:
        if np.dot(chrom, self.weights) > self.capacity:
            return False
        for i, j in self.conflicts:
            if chrom[i] and chrom[j]:
                return False
        return True

    def weight(self, chrom: np.ndarray) -> int:
        return int(np.dot(chrom, self.weights))

    def value(self, chrom: np.ndarray) -> int:
        return int(np.dot(chrom, self.values))

    def repair(self, chrom: np.ndarray, rng: random.Random) -> np.ndarray:
        """
        Ремонт:
        1. Устраняем конфликты: для каждой конфликтующей пары оставляем предмет
           с большим value/weight (или случайный при равенстве).
        2. Если вес всё ещё превышает capacity — удаляем предметы
           с наименьшим value/weight, пока не уложимся.
        """
        x = chrom.copy()

        # 1. Устранение конфликтов
        for i, j in self.conflicts:
            if x[i] and x[j]:
                ratio_i = self.values[i] / self.weights[i]
                ratio_j = self.values[j] / self.weights[j]
                if ratio_i > ratio_j:
                    x[j] = 0
                elif ratio_j > ratio_i:
                    x[i] = 0
                else:
                    # равные — случайный выбор
                    if rng.random() < 0.5:
                        x[i] = 0
                    else:
                        x[j] = 0

        # 2. Восстановление по весу
        current_w = self.weight(x)
        if current_w <= self.capacity:
            return x

        # Сортируем выбранные предметы по возрастанию value/weight
        selected = np.where(x == 1)[0]
        ratios = [(self.values[i] / self.weights[i], i) for i in selected]
        ratios.sort()  # сначала самые «плохие»

        for _, i in ratios:
            if current_w <= self.capacity:
                break
            x[i] = 0
            current_w -= self.weights[i]

        return x

    def penalty_fitness(self, chrom: np.ndarray, penalty_coef: float = 50.0) -> float:
        """
        Штрафная функция:
        fitness = value - penalty_coef * (перевес + число нарушенных конфликтов)
        """
        val = self.value(chrom)
        overweight = max(0, self.weight(chrom) - self.capacity)
        conflict_viol = sum(1 for i, j in self.conflicts if chrom[i] and chrom[j])
        return val - penalty_coef * (overweight + 10 * conflict_viol)


class GeneticAlgorithm:
    def __init__(
        self,
        problem: KnapsackConflicts,
        pop_size: int = 80,
        generations: int = 150,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.02,
        tournament_size: int = 3,
        elitism: int = 2,
        constraint_mode: str = "repair",   # "repair" | "penalty"
        crossover_type: str = "one_point", # "one_point" | "uniform"
        penalty_coef: float = 50.0,
        seed: int | None = None,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.constraint_mode = constraint_mode
        self.crossover_type = crossover_type
        self.penalty_coef = penalty_coef
        self.rng = random.Random(seed)
        self.n = problem.n

        # История лучшего значения
        self.best_history: List[float] = []
        self.avg_history: List[float] = []

    def _random_chrom(self) -> np.ndarray:
        return np.array([self.rng.randint(0, 1) for _ in range(self.n)], dtype=int)

    def _init_population(self) -> List[np.ndarray]:
        pop = []
        for _ in range(self.pop_size):
            chrom = self._random_chrom()
            if self.constraint_mode == "repair":
                chrom = self.problem.repair(chrom, self.rng)
            pop.append(chrom)
        return pop

    def _fitness(self, chrom: np.ndarray) -> float:
        if self.constraint_mode == "repair":
            # после ремонта все особи допустимы → fitness = value
            return float(self.problem.value(chrom))
        else:
            return self.problem.penalty_fitness(chrom, self.penalty_coef)

    def _tournament(self, pop: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        idxs = self.rng.sample(range(len(pop)), self.tournament_size)
        best_idx = max(idxs, key=lambda i: fitnesses[i])
        return pop[best_idx].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.crossover_rate:
            return p1.copy(), p2.copy()

        if self.crossover_type == "one_point":
            point = self.rng.randint(1, self.n - 1)
            c1 = np.concatenate([p1[:point], p2[point:]])
            c2 = np.concatenate([p2[:point], p1[point:]])
        else:  # uniform
            mask = np.array([self.rng.random() < 0.5 for _ in range(self.n)])
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
        return c1, c2

    def _mutate(self, chrom: np.ndarray) -> np.ndarray:
        for i in range(self.n):
            if self.rng.random() < self.mutation_rate:
                chrom[i] = 1 - chrom[i]
        return chrom

    def run(self) -> Dict[str, Any]:
        population = self._init_population()
        fitnesses = [self._fitness(ind) for ind in population]

        best_ever = None
        best_ever_fit = -np.inf

        for gen in range(self.generations):
            # Элитизм
            elite_idxs = np.argsort(fitnesses)[-self.elitism:][::-1]
            elites = [population[i].copy() for i in elite_idxs]

            new_pop = elites[:]

            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                if self.constraint_mode == "repair":
                    c1 = self.problem.repair(c1, self.rng)
                    c2 = self.problem.repair(c2, self.rng)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = new_pop[:self.pop_size]
            fitnesses = [self._fitness(ind) for ind in population]

            # Статистика
            best_fit = max(fitnesses)
            avg_fit = float(np.mean(fitnesses))
            self.best_history.append(best_fit)
            self.avg_history.append(avg_fit)

            if best_fit > best_ever_fit:
                best_ever_fit = best_fit
                best_idx = int(np.argmax(fitnesses))
                best_ever = population[best_idx].copy()

        # Финальная проверка допустимости (на всякий случай)
        if self.constraint_mode == "penalty" and best_ever is not None:
            # Для penalty-режима применяем ремонт к лучшему найденному,
            # чтобы вернуть допустимое решение
            best_ever = self.problem.repair(best_ever, self.rng)
            best_ever_fit = float(self.problem.value(best_ever))

        return {
            "best_chromosome": best_ever,
            "best_value": best_ever_fit,
            "best_weight": self.problem.weight(best_ever) if best_ever is not None else 0,
            "is_feasible": self.problem.is_feasible(best_ever) if best_ever is not None else False,
            "selected_items": [i for i, bit in enumerate(best_ever) if bit] if best_ever is not None else [],
            "best_history": self.best_history,
            "avg_history": self.avg_history,
        }


# -------------------- Базовые эвристики --------------------

def greedy_heuristic(problem: KnapsackConflicts) -> Dict[str, Any]:
    """
    Жадный алгоритм: сортируем по value/weight убыванию,
    добавляем, если не нарушает вес и конфликты.
    """
    ratios = [(problem.values[i] / problem.weights[i], i) for i in range(problem.n)]
    ratios.sort(reverse=True)

    chrom = np.zeros(problem.n, dtype=int)
    current_w = 0
    selected = set()

    for _, i in ratios:
        # проверка конфликтов
        if any(j in selected for j in problem.conflict_sets[i]):
            continue
        if current_w + problem.weights[i] <= problem.capacity:
            chrom[i] = 1
            selected.add(i)
            current_w += problem.weights[i]

    return {
        "chromosome": chrom,
        "value": problem.value(chrom),
        "weight": problem.weight(chrom),
        "selected_items": list(selected),
        "is_feasible": True,
    }


def random_feasible_search(problem: KnapsackConflicts, n_trials: int = 5000, seed: int = 123) -> Dict[str, Any]:
    """
    Случайный поиск допустимых решений (с ремонтом).
    """
    rng = random.Random(seed)
    best_val = -1
    best_chrom = None

    for _ in range(n_trials):
        chrom = np.array([rng.randint(0, 1) for _ in range(problem.n)], dtype=int)
        chrom = problem.repair(chrom, rng)
        val = problem.value(chrom)
        if val > best_val:
            best_val = val
            best_chrom = chrom.copy()

    return {
        "chromosome": best_chrom,
        "value": best_val,
        "weight": problem.weight(best_chrom),
        "selected_items": [i for i, b in enumerate(best_chrom) if b],
        "is_feasible": True,
        "n_trials": n_trials,
    }
