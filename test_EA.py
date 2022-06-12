import math

from EvolutionAlgorithm import Evolution
import cec2017.functions as func


def square_function(x):
    return x[0] ** 2 + x[1] ** 2


def f2(x):
    e = math.e
    return 1.5 - (e ** (-x[0] ** 2 - x[1] ** 2)) - 0.5 * (e ** (-(x[0] - 1) ** 2 - (x[1] + 2) ** 2))


def main():
    test_ea()


def test_ea():
    ea1 = Evolution(func.f9, 10, 100, 100, 100, False, 1, 0.8)
    ea1.learn()
    points = ea1.get_points_for_approximator(2, True)
    ea1.plot_population_move(only_best_points=False)
    ea1.plot_best_points_values_history()
    ea1.plot_generations_means()


if __name__ == '__main__':
    main()
