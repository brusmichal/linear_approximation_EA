import math

from EvolutionAlgorithm import Evolution
import cec2017.functions as func


def square_function(x):
    return x[0] ** 2 + x[1] ** 2


def f2(x):
    e = math.e
    return 1.5 - (e ** (-x[0] ** 2 - x[1] ** 2)) - 0.5 * (e ** (-(x[0] - 1) ** 2 - (x[1] + 2) ** 2))


def main():
    ea1 = Evolution(func.f3, 10, 100, 20, 10000, True, 0.1, 0.1, 0.8)
    ea1.learn()
    ea1.plot_means()
    ea1.plot_best_history()
    # ea1.plot_steps()
    ea1.plot_steps_with_contour_plot()


if __name__ == '__main__':
    main()
