import numpy as np


class Evolution:
    def __init__(self, goal_function, function_dimension, population_size, max_iter, p_mutation, mutation_strength,
                 p_crossover):
        self.tournament_size = 2
        self.goal_function = goal_function
        self.population_dim = function_dimension
        self.dim_upper_bound = 100
        self.population_size = population_size
        self.current_gen_count = 0
        self.max_gen_count = max_iter
        self.p_mutation = p_mutation
        self.mutation_strength = mutation_strength
        self.p_crossover = p_crossover
        self.current_population = np.empty([self.population_size, self.population_dim])
        self.current_evaluation = np.empty([self.population_size])
        self.population_history = None
        self.current_best = None
        self.best_overall = None
        self.best_history = None

    def start(self):
        self.initialize_population()
        while self.current_gen_count < self.max_gen_count:
            self.next_generation()

    def next_generation(self):
        reproduced = self.reproduction()
        mutants = self.genetic_operations(reproduced)
        evaluation = self.evaluate(mutants)
        best = self.find_best(mutants, evaluation)
        if self.best_overall[1] < best[1]:
            self.best_overall = best
        self.succession()

    def initialize_population(self):
        rng = np.random.default_rng()
        self.current_population = [rng.uniform(-self.dim_upper_bound, self.dim_upper_bound, size=self.population_dim) \
                                   for _ in range(self.population_size)]
        self.current_evaluation = self.evaluate(self.current_population)
        self.current_best = self.find_best(self.current_population, self.current_evaluation)

    def evaluate(self, population):
        evaluation = [self.goal_function(individual) for individual in population]
        return evaluation

    def find_best(self, population, evaluation):
        best = [population[np.argmax(evaluation)], np.max(evaluation)]
        return best

    def reproduction(self):
        return self.tournament_selection()

    def genetic_operations(self, reproduced):

        return NotImplemented

    def succession(self):
        return NotImplemented

    def tournament_selection(self):
        selected = np.array([self.population_size])
        sorted_population = sorted(self.current_population, key=lambda individual: individual[1])
        for i in range(self.population_size):
            rng = np.random.default_rng()
            fighters = rng.choice(sorted_population, size=self.tournament_size)
            winner = fighters[np.argmax(fighters, axis=1)]
            selected[i] = winner
        return selected
