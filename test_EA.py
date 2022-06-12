import math
import numpy as np
from pprint import pprint
from EvolutionAlgorithm import Evolution
import cec2017.functions as cec


def square_function(x):
    return x[0] ** 2 + x[1] ** 2


def f2(x):
    e = math.e
    return 1.5 - (e ** (-x[0] ** 2 - x[1] ** 2)) - 0.5 * (e ** (-(x[0] - 1) ** 2 - (x[1] + 2) ** 2))


def main():
    # test_ea()
    make_statistics(1)


def make_statistics(runs_number):
    populations = np.array([20, 50, 75, 100, 125])
    mutations = np.array([0.05, 0.1, 0.5, 1, 2])
    means = np.empty((len(cec.all_functions), len(populations), len(mutations)))
    stds = np.empty((len(cec.all_functions), len(populations), len(mutations)))
    print(f"Populations: {populations}")
    print(f"Mutation strengths: {mutations}\n")
    for l, function in enumerate(cec.all_functions):
        print(f"Function {function}:\n")
        for k, population in enumerate(populations):
            # print(f"Population size: {population}")
            for j, mutation in enumerate(mutations):
                # print(f"Mutation strength: {mutation}")
                minimums = np.empty([runs_number])
                for i in range(runs_number):
                    ea = Evolution(function, 10, 100, population, 20000 / population, False, mutation, 0.8)
                    ea.learn()
                    minimum = ea.get_optimum()[1]
                    minimums[i] = minimum
                means[l][k][j] = minimums.mean()
                stds[l][k][j] = minimums.std()
        print("Means:")
        print(np.array_str(means[l], precision=2))
        print("Standard deviations:")
        print(np.array_str(stds[l], precision=2))
        best_params = np.unravel_index(means[l].argmin(), means[l].shape)
        print(f"Best parameters position: {best_params[0], best_params[1]}")
        print(f"Best parameters values (pop_size, mut_str): {populations[best_params[0]], mutations[best_params[1]]}")
        print("--------------------------------------------------------------------------------------------------")



def test_ea():
    ea1 = Evolution(cec.f1, 10, 100, 20, 10000, True, 0.1, 0.8)
    ea1.learn()
    points = ea1.get_points_for_approximator(2, True)
    ea1.plot_population_move(only_best_points=False)
    ea1.plot_best_points_values_history()
    ea1.plot_generations_means()


if __name__ == '__main__':
    main()
