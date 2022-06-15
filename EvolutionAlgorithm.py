import numpy as np
import matplotlib.pyplot as plt


class Evolution:
    def __init__(self, goal_function, function_dimension, upper_bound, population_size, max_iter, with_crossing,
                 mutation_strength,
                 p_crossover):
        self.tournament_size = 2
        self.goal_function = goal_function
        self.population_dim = function_dimension
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.current_gen_count = 0
        self.max_gen_count = max_iter
        self.with_crossing = with_crossing
        self.mutation_strength = mutation_strength
        self.p_crossover = p_crossover
        self.current_population = np.empty([self.population_size, self.population_dim])
        self.current_evaluation = np.empty([self.population_size])
        self.current_best = None
        self.best_overall = None
        self.evaluation_history = []
        self.population_history = []
        self.best_history = []

    def get_optimum(self):
        return self.best_overall

    def get_last_population_with_evaluation(self):
        eval_pop = [np.append(x, y) for x, y in list(zip(self.current_population, self.current_evaluation))]
        return eval_pop

    def get_points_for_approximator(self, k_th_generation, up_to):
        size = len(self.population_history)* self.population_size
        points = np.zeros([size,self.population_dim+2], dtype=float)
        j = 0
        for i, (population, evaluation) in enumerate(zip(self.population_history, self.evaluation_history)):
            eval_pop = [np.append(np.append([1], x), y) for x, y in list(zip(population, evaluation))]
            for point in eval_pop:
                points[j] = point
                j+=1
        if up_to:
            return points[:k_th_generation*self.population_size]
        else:
            return points[k_th_generation*self.population_size]

    def run(self):
        self.initialize_population()
        while self.current_gen_count < self.max_gen_count:
            self.next_generation()
            self.current_gen_count += 1

    def next_generation(self):
        reproduced = self.reproduction()
        mutants = self.genetic_operations(reproduced)
        evaluation = self.evaluate(mutants)
        best = self.find_best(mutants, evaluation)
        if best[1] < self.best_overall[1]:
            self.best_overall = best

        self.population_history.append(self.current_population)
        self.evaluation_history.append(self.current_evaluation)
        self.best_history.append(self.current_best)

        self.current_population, self.current_evaluation, self.current_best \
            = self.succession(mutants, evaluation)

    def initialize_population(self):
        rng = np.random.default_rng()
        self.current_population = np.array(
            [rng.uniform(-self.upper_bound, self.upper_bound, size=self.population_dim) \
             for _ in range(self.population_size)])
        self.current_evaluation = self.evaluate(self.current_population)
        self.current_best = self.best_overall = self.find_best(self.current_population, self.current_evaluation)

    def evaluate(self, population):
        evaluation = np.array([self.goal_function(individual) for individual in population])
        return evaluation

    def find_best(self, population, evaluation):
        best = population[np.argmin(evaluation)], np.array([np.min(evaluation)])
        return best

    def reproduction(self):
        return self.tournament_selection()

    def genetic_operations(self, reproduced):
        children = self.arithmetic_crossover(reproduced)
        mutants = self.mutation_gaussian(children)
        return mutants

    def succession(self, mutants, evaluation):
        population = np.concatenate((mutants, self.current_best[0].reshape((1, self.population_dim))), axis=0)
        evaluation = np.concatenate((evaluation, self.current_best[1]))
        pop_eval = list(zip(population, evaluation))
        sorted_pop = sorted(pop_eval, key=lambda x: x[1])
        sorted_pop = sorted_pop[:-1]
        population = np.array([x[0] for x in sorted_pop])
        evaluation = np.array([x[1] for x in sorted_pop])
        return population, evaluation, self.find_best(population, evaluation)

    def tournament_selection(self):
        selected = np.empty([self.population_size, self.population_dim])
        for i in range(self.population_size):
            rng = np.random.default_rng()
            fighters = rng.choice(self.current_population, size=self.tournament_size)
            fighters_eval = np.array([self.goal_function(x) for x in fighters])
            winner = fighters[np.argmin(fighters_eval)]
            selected[i] = winner
        return selected

    def arithmetic_crossover(self, reproduced):
        children = np.empty([self.population_size, self.population_dim])
        rng = np.random.default_rng()
        for i in range(0, self.population_size - 1, 2):
            parent_a, parent_b = rng.choice(reproduced, 2)
            weights_a = rng.uniform(0, 1, size=reproduced.shape[1])
            weights_b = rng.uniform(0, 1, size=reproduced.shape[1])
            if rng.uniform(0, 1) < self.p_crossover:
                child_a = np.empty([self.population_dim])
                child_b = np.empty([self.population_dim])
                # whole arithmetic recombination
                for j in range(len(weights_a)):
                    child_a = weights_a[j] * parent_a + (1 - weights_a[j]) * parent_b
                    child_b = weights_a[j] * parent_a + (1 - weights_b[j]) * parent_b
                children[i] = child_a
                children[i + 1] = child_b
            else:
                children[i] = parent_a
                children[i + 1] = parent_b
        return children

    def mutation_gaussian(self, children):
        rng = np.random.default_rng()
        for i in range(len(children)):
            mutation_matrix = rng.normal(0, 1, size=len(children[i]))
            children[i] = children[i] + self.mutation_strength * mutation_matrix
        return children

    def plot_best_points_values_history(self):
        y = np.array([x[1] for x in self.best_history])
        x = np.arange(self.max_gen_count)
        plt.figure(figsize=(20, 10))
        plt.plot(x, y)
        plt.yscale('log')
        plt.title(
            f"Wartość funkcji dla najlepszego punktu w danej generacji. Wartość w optimum = {self.best_overall[1]}")
        plt.xlabel("Generacja")
        plt.ylabel("q(x_best)")
        plt.grid(b=True)
        plt.show()

    def plot_steps_2d(self):
        x = np.array(self.population_history).flatten()
        y = np.array(self.evaluation_history).flatten()
        x1 = np.linspace(-5, 5, 100)
        y1 = [self.goal_function(x_) for x_ in x1]
        plt.figure(figsize=(20, 10))
        plt.plot(x, y)
        plt.plot(x1, y1)
        plt.title("Kolejne najlepsze punkty")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(b=True)
        plt.show()



    def plot_generations_means(self):
        y = np.array([evaluation.mean() for evaluation in self.evaluation_history])
        x = np.arange(self.max_gen_count)
        plt.figure(figsize=(20, 10))
        plt.plot(x, y)
        plt.yscale('log')
        plt.title(f"Wartość średnia funkcji dla populacji w danej generacji. Dla ostatniej populacji: {y[-1]}")
        plt.xlabel("Generacja")
        plt.ylabel("E(q(x))")
        plt.grid(b=True)
        plt.show()

    def plot_population_move(self, only_best_points: bool):
        plt.figure(figsize=(40, 20))
        x = np.arange(-self.upper_bound, self.upper_bound, 0.5)
        y = np.arange(-self.upper_bound, self.upper_bound, 0.5)
        X, Y = np.meshgrid(x, y)
        Z = np.empty(X.shape)
        for k in range(X.shape[0]):
            for l in range(X.shape[1]):
                Z[k, l] = self.goal_function(np.array([X[k, l], Y[k, l]]))

        plt.contour(X, Y, Z, 20)
        minimum = self.best_overall
        if only_best_points:
            steps = np.array([x[0] for x in self.best_history])
            for m in range(len(steps[:-1])):
                plt.arrow(steps[m][0], steps[m][1], steps[m + 1][0] - steps[m][0], steps[m + 1][1] - steps[m][1])
        else:
            steps = np.array(self.population_history)
            steps = steps.reshape(-1, steps.shape[-1])
            plt.scatter(steps[:, 0], steps[:, 1])
        if only_best_points:
            plt.title(f"Historia najlepszych punktów. Minimum = {minimum[1]}")
        else:
            plt.title(f"Historia populacji. Minimum = {minimum[1]}")
        plt.plot(minimum[0][0], minimum[0][1], '*')
        plt.show()
