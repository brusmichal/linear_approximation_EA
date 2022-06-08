from EvolutionAlgorithm import Evolution


def main():
    def square_function(x):
        return x ** 2

    ea1 = Evolution(square_function, 1, 20, 100, False, 0.1, 0.1, 0.8)
    ea1.learn()
    ea1.plot_means()
    ea1.plot_best_history()
    ea1.plot_steps()


if __name__ == '__main__':
    main()
