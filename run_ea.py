import math
import numpy as np
import pandas as pd
from EvolutionAlgorithm import Evolution
import cec2017.functions as cec


def square_function(x):
    return x[0] ** 2 + x[1] ** 2


def f(x):
    e = math.e
    return 1.5 - (e ** (-x[0] ** 2 - x[1] ** 2)) - 0.5 * (e ** (-(x[0] - 1) ** 2 - (x[1] + 2) ** 2))


def make_params_statistics(runs_number, with_print):
    populations = np.array([20, 50, 80, 100, 120, 150])
    mutations = np.array([0.05, 0.1, 0.5, 1, 2, 5])
    means = np.empty((len(cec.all_functions), len(populations), len(mutations)))
    stds = np.empty((len(cec.all_functions), len(populations), len(mutations)))
    if with_print:
        print(f"Populations: {populations}")
        print(f"Mutation strengths: {mutations}\n")
    for l, function in enumerate(cec.all_functions):
        if with_print:
            print(f"Function {function}:\n")
        for k, population in enumerate(populations):

            # print(f"Population size: {population}")
            for j, mutation in enumerate(mutations):
                # print(f"Mutation strength: {mutation}")
                minimums = np.empty([runs_number])
                for i in range(runs_number):
                    ea = Evolution(function, 10, 100, population, 10000 / population, True, mutation, 0.8)
                    ea.run()
                    minimum = ea.get_optimum()[1]
                    minimums[i] = minimum
                means[l][k][j] = minimums.mean()
                stds[l][k][j] = minimums.std()
        if with_print:
            print("Means:")
            print(np.array_str(means[l], precision=2))
            print("Standard deviations:")
            print(np.array_str(stds[l], precision=2))
            best_params = np.unravel_index(means[l].argmin(), means[l].shape)
            print(f"Best parameters position: {best_params[0], best_params[1]}")
            print(
                f"Best parameters values (pop_size, mut_str): {populations[best_params[0]], mutations[best_params[1]]}")
            print(f"Mean minimum with these params: {means[l][best_params[0]][best_params[1]]}")
            print("--------------------------------------------------------------------------------------------------")
        pd.DataFrame(means[l]).to_csv('stats/ae/means/means' + str(l) + '.csv')
        pd.DataFrame(stds[l]).to_csv('stats/ae/stds/stds' + str(l) + '.csv')


def test_example_function(func):
    ea1 = Evolution(func, 10, 100, 20, 5000, True, 0.1, 0.8)
    ea1.run()
    ea1.plot_population_move(only_best_points=False)
    ea1.plot_population_move(only_best_points=True)
    ea1.plot_best_points_values_history()
    ea1.plot_generations_means()


def main():
    test_example_function(cec.f4)
    # make_params_statistics(25, with_print=True)


if __name__ == '__main__':
    main()
