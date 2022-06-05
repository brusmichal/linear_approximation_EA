import numpy as np


class Evolution:
    def __init__(self, goal_function, function_dimension, population_size, max_iter, with_crossing, p_mutation, mutation_strength,
                 p_crossover):
        self.tournament_size = 2
        self.goal_function = goal_function
        self.population_dim = function_dimension
        self.dim_upper_bound = 100
        self.population_size = population_size
        self.current_gen_count = 0
        self.max_gen_count = max_iter
        self.with_crossing = with_crossing
        self.p_mutation = p_mutation
        self.mutation_strength = mutation_strength
        self.p_crossover = p_crossover
        self.current_population = np.empty([self.population_size, self.population_dim])
        self.current_evaluation = np.empty([self.population_size])
        self.current_best = None
        self.best_overall = None
        self.evaluation_history = []
        self.population_history = []
        self.best_history = []

    def learn(self):
        self.initialize_population()
        while self.current_gen_count < self.max_gen_count:
            self.next_generation()
            self.current_gen_count += 1

    def next_generation(self):
        reproduced = self.reproduction()
        mutants = self.genetic_operations(reproduced)
        evaluation = self.evaluate(mutants)
        best = self.find_best(mutants, evaluation)
        if self.best_overall[1] < best[1]:
            self.best_overall = best

        self.population_history.append(self.current_population)
        self.evaluation_history.append(self.current_evaluation)
        self.best_history.append(self.current_best)

        self.current_population, self.current_evaluation, self.current_best \
            = self.succession(mutants, evaluation)

    def initialize_population(self):
        rng = np.random.default_rng()
        self.current_population = [rng.uniform(-self.dim_upper_bound, self.dim_upper_bound, size=self.population_dim) \
                                   for _ in range(self.population_size)]
        self.current_evaluation = self.evaluate(self.current_population)
        self.current_best = self.best_overall = self.find_best(self.current_population, self.current_evaluation)

    def evaluate(self, population):
        evaluation = np.array([self.goal_function(individual) for individual in population])
        return evaluation

    def find_best(self, population, evaluation):
        best = population[np.argmax(evaluation)], np.array([np.max(evaluation)])
        return best

    def reproduction(self):
        return self.tournament_selection()

    def genetic_operations(self, reproduced):
        children = self.crossover(reproduced)
        mutants = self.mutation(children)
        return mutants

    def succession(self, mutants, evaluation):
        population = np.concatenate((mutants, self.current_best[0].reshape((self.population_dim, 1))))
        evaluation = np.concatenate((evaluation, self.current_best[1].reshape((1, 1))))
        pop_eval = np.append(population, evaluation, axis=1)
        sorted_pop = sorted(pop_eval, key=lambda x: x[1])
        sorted_pop = sorted_pop[:-1, :]
        return sorted_pop[0], sorted_pop[1], self.find_best(sorted_pop[0], sorted_pop[1])

    def tournament_selection(self):
        selected = np.empty([self.population_size, self.population_dim])
        for i in range(self.population_size):
            rng = np.random.default_rng()
            fighters = rng.choice(self.current_population, size=self.tournament_size)
            winner = fighters[np.argmax(fighters)]
            selected[i] = winner
        return selected

    def crossover(self, reproduced):
        children = np.empty([self.population_size, self.population_dim])
        rng = np.random.default_rng()
        for i in range(int(reproduced.shape[0] / 2)):
            parent_a, parent_b = rng.choice(reproduced, 2)
            weights_a = rng.uniform(0, 1, size=reproduced.shape[1])
            weights_b = rng.uniform(0, 1, size=reproduced.shape[1])
            if rng.uniform(0, 1) < self.p_crossover:
                child_a = np.empty([self.population_dim])
                child_b = np.empty([self.population_dim])
                for j in range(len(weights_a)):
                    child_a[j] = weights_a[j] * parent_a[j] + (1 - weights_a[j]) * parent_b[j]
                    child_b[j] = weights_a[j] * parent_a[j] + (1 - weights_b[j]) * parent_b[j]
                children[i] = child_a
                children[i+1] = child_b
            else:
                children[i] = parent_a
                children[i + 1] = parent_b
        return children

    def mutation(self, children):
        rng = np.random.default_rng()
        for child in children:
            if rng.uniform(0, 1) < self.p_mutation:
                mutation_matrix = rng.normal(0, 1, size=len(child))
                child = child + self.mutation_strength * mutation_matrix
        return children

    def plot_best_history(self):
        x = self.best_history
        y = np.arange(self.max_gen_count)
        plt.figure(figsize=(20, 10))
        plt.scatter(x, y)
        plt.title("Wartość najlepszego punktu w danej generacji")
        plt.xlabel("Generacje")
        plt.ylabel("q(x_best)")
        plt.grid(b=True)
        plt.show()

    def plot_means(self):
        x = [self.evaluation_history[i].mean() for i in range(self.max_gen_count)]
        y = np.arange(self.max_gen_count)
        plt.figure(figsize=(20, 10))
        plt.scatter(x, y)
        plt.title("Wartość średnia populacji w danej generacji")
        plt.xlabel("Generacje")
        plt.ylabel("E(q(x))")
        plt.grid(b=True)
        plt.show()
