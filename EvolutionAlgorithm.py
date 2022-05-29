import numpy as np


class Evolution:
    def __init__(self, goal_function, function_dimension, population_size, max_iter, p_mutation, mutation_strength, p_crossover):
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
        self.evaluate_population()
        while self.current_gen_count < self.max_gen_count:
            self.next_generation()

    def next_generation(self):
        self.reproduction()
        self.genetic_operations()
        self.evaluate_population()
        if self.best_overall[1] < self.current_best[1]:
            self.best_overall = self.current_best
        self.succession()

    def initialize_population(self):
        rng = np.random.default_rng()
        self.current_population = [rng.uniform(-self.dim_upper_bound, self.dim_upper_bound, size=self.population_dim) \
                                   for _ in range(self.population_size)]

    def evaluate_population(self):
        self.current_evaluation = [self.goal_function(individual) for individual in self.current_population]

        self.current_best = [self.current_population[np.argmax(self.current_evaluation)], np.max(self.current_evaluation)]

    def reproduction(self):
        pass

    def genetic_operations(self):
        pass

    def succession(self):
        pass
