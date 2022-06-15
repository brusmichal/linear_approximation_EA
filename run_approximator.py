from Approximator import Approximator
from EvolutionAlgorithm import Evolution
import cec2017.functions as cec
import numpy as np
import pandas as pd


def x_square(x):
    sum = 0
    for n in range(x.shape[0]):
        sum += x[n] ** 2
    return sum


# tylko dla wymiarów 1 i 2
def matlab_test(k_list, a_list, domain_dim):
    lines = []
    for m in range(len(k_list)):
        lines.append('x_' + str(m) + '= [')
        if domain_dim == 2:
            for n in range(k_list[m].shape[0]):
                lines.append(str(k_list[m][n][1:k_list[m].shape[1] - 1][0]) +
                             " " + str(k_list[m][n][1:k_list[m].shape[1] - 1][1]))
        elif domain_dim == 1:
            for n in range(k_list[m].shape[0]):
                lines.append(str(k_list[m][n][1:k_list[m].shape[1] - 1][0]))
        lines.append('];')
        lines.append('y_' + str(m) + '= [')
        for n in range(k_list[m].shape[0]):
            lines.append(str(k_list[m][n][k_list[m].shape[1] - 1]))
        lines.append('];')
        lines.append('b_' + str(m) + '=' + str(a_list[m][0][0]) + ';')
        lines.append('a_' + str(m) + '=' + str(a_list[m][1][0]) + ';')
    if domain_dim == 1:
        lines.append('hold on')
        for m in range(len(k_list)):
            lines.append('plot(x_' + str(m) + ',x_' + str(m) + '.^2,"." )')
            lines.append('plot(x_' + str(m) + ',x_' + str(m) +
                         ' .*a_' + str(m) + '+b_' + str(m) + ')')
        lines.append('hold off')
    elif domain_dim == 2:
        lines.append('plot3(...')
        for m in range(len(k_list) - 1):
            lines.append('x_' + str(m) + '(:,1),x_' + str(m) +
                         '(:,2),y_' + str(m) + ',...')  # \'.\',
        m = len(k_list) - 1
        lines.append('x_' + str(m) + '(:,1),x_' + str(m) + '(:,2),y_' + str(m) + ')')

    with open('matlab_test.m', 'w') as f:
        f.write('\n'.join(lines))


def iterate_through_generations(generations_num, fun_list, domain_dim, upp_bound, population_size):
    for n in range(len(fun_list)):
        try:
            print(f'Function number: {n + 3}')
            results = []
            fun = fun_list[n]
            ea1 = Evolution(goal_function=fun, function_dimension=domain_dim, upper_bound=upp_bound,
                            population_size=population_size, max_iter=generations_num,
                            with_crossing=False, mutation_strength=0.5, p_crossover=0.8)
            ea1.run()
            points = ea1.get_points_for_approximator(generations_num, True)
            model_best_min = 10e20
            ae_best_min = 10e20
            for gen_num in range(1000, generations_num, 1000):
                try:
                    # print((gen_num))#*population_size)
                    pts = points[0:(gen_num + 1) * population_size]
                    # print(pts.size)
                    approx = Approximator(UPPER_BOUND=upp_bound, DIMENTIONALITY=domain_dim, limit_err=5)
                    approx.K_list.append(pts)
                    a = approx.calculate_a_vector(pts)
                    approx.a_list.append(a)
                    a_list, k_list = approx.check_domain()
                    model_minimum = approx.find_minimum(function=fun)[0]
                    ea_minimum = approx.find_minimum_in_init_data()[0]
                    results.append([model_minimum, ea_minimum])
                    print([model_minimum, ea_minimum])
                except ValueError:
                    print('Value error!!!')
                    pass
                except np.linalg.LinAlgError:
                    print("Singular matrix!!!")
                    pass
        except KeyboardInterrupt:
            m_bett, m_best_min, ae_best_min = show_results(results, model_best_min, ae_best_min)
            print(f'Model was better in {m_bett} of {len(results)} attempts')
            print(f'Model best min was {m_best_min}, and EA best min was: {ae_best_min}')
            break
        m_bett, m_best_min, ae_best_min = show_results(results, model_best_min, ae_best_min)
        print(f'Model was better in {m_bett} of {len(results)} attempts')
        print(f'Model best min was {m_best_min}, and EA best min was: {ae_best_min}')


def make_statistics(runs_number, with_print):
    functions_list = cec.all_functions[16, ]
    generations_number = np.array([10, 25, 50, 75, 100])
    populations_for_func = np.array(
        [20, 20, 20, 20, 150, 50, 120, 120, 20, 120, 20, 50, 50, 20, 20, 150, 80, 120, 20, 80, 100, 20, 80, 120, 20, 50,
         100, 20, 150, 20])
    mutations_for_func = np.array(
        [1, 2, 2, 5, 1, 5, 2, 1, 2, 2, 5, 2, 0.05, 5, 5, 5, 0.5, 0.5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2])
    means = np.empty([len(cec.all_functions), len(generations_number), 3])
    stds = np.empty([len(cec.all_functions), len(generations_number), 2])
    for l, function in enumerate(functions_list):
        if l < -1:  ### tutaj można na chama utawić ok jakiego miejsca listy funkcji ma liczyć
            continue
        if with_print:
            print(f"Function {function}:\n")
        population = populations_for_func[l]
        mutation = mutations_for_func[l]
        minimums = np.empty([len(generations_number), 2, runs_number])
        for i in range(runs_number):
            ea = Evolution(function, 10, 100, population, generations_number[-1], True, mutation, 0.8)
            ea.run()
            points = ea.get_points_for_approximator(generations_number[-1], True)
            print(f' points: {len(points)}')
            for j, gen_num in enumerate(generations_number):
                for attempt in range(20):
                    try:
                        pts = points[0:gen_num]
                        print(len(pts))
                        approx = Approximator(UPPER_BOUND=100, DIMENTIONALITY=10, limit_err=10)
                        approx.K_list.append(pts)
                        a = approx.calculate_a_vector(pts)
                        approx.a_list.append(a)
                        approx.check_domain()
                        model_minimum = approx.find_minimum(function=function)[0]
                        ea_minimum = approx.find_minimum_in_init_data()[0]
                        minimums[j][0][i] = model_minimum
                        minimums[j][1][i] = ea_minimum
                    except np.linalg.LinAlgError:
                        print("Singular matrix!!!")
                        pass
                    else:
                        break

        for generation in range(len(generations_number)):
            model_was_better = 0
            for run in range(runs_number):
                if minimums[generation][0][run] < minimums[generation][1][run]:
                    model_was_better += 1
            model_gen_mean = minimums[generation][0][:]
            ea_gen_mean = minimums[generation][1][:]
            means[l][generation][0] = round(model_gen_mean.mean(), 2)  # approx
            means[l][generation][1] = round(ea_gen_mean.mean(), 2)  # ea
            means[l][generation][2] = round((model_was_better / runs_number) * 100, 2)  # ea
            stds[l][generation][0] = round(model_gen_mean.std(), 2)
            stds[l][generation][1] = round(ea_gen_mean.std(), 2)
            # print(minimums)
        if with_print:
            print(means)
        pd.DataFrame(means[l]).to_csv('stats/means/means' + str(l + 1) + '.csv')
        pd.DataFrame(stds[l]).to_csv('stats/stds/stds' + str(l + 1) + '.csv')


def show_results(results, model_best_min, ae_best_min):
    model_is_better = 0
    for element in results:
        # print(element)
        if element[0] < element[1]:
            model_is_better += 1
        if element[0] < model_best_min:
            model_best_min = element[0]
        if element[1] < model_best_min:
            ae_best_min = element[1]
    return model_is_better, model_best_min, ae_best_min


def main():
    # fun_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    fun_list = [cec.f8]
    domain_dim = 10
    upp_bound = 100
    population_size = 10
    generations_num = 6000
    # iterate_through_generations(generations_num, fun_list, domain_dim, upp_bound, population_size)
    make_statistics(25, True)


if __name__ == '__main__':
    main()

# matlab_test(k_list, a_list, domain_dim)
