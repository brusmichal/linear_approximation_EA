from Approximator import Approximator
from EvolutionAlgorithm import Evolution
from cec2017.functions import *

def x_square(x):
    sum = 0
    for n in range(x.shape[0]):
        sum += x[n]**2
    return sum

#tylko dla wymiarów 1 i 2
def matlab_test(k_list, a_list, domain_dim):
    lines = []
    for m in range(len(k_list)):
        lines.append('x_'+str(m)+'= [')
        if domain_dim == 2:
            for n in range(k_list[m].shape[0]):
                lines.append(str(k_list[m][n][1:k_list[m].shape[1]-1][0]) +
                            " "+str(k_list[m][n][1:k_list[m].shape[1]-1][1]))
        elif domain_dim ==1:
            for n in range(k_list[m].shape[0]):
                lines.append(str(k_list[m][n][1:k_list[m].shape[1]-1][0]))
        lines.append('];')
        lines.append('y_'+str(m)+'= [')
        for n in range(k_list[m].shape[0]):
            lines.append(str(k_list[m][n][k_list[m].shape[1]-1]))
        lines.append('];')
        lines.append('b_'+str(m)+'='+str(a_list[m][0][0])+';')
        lines.append('a_'+str(m)+'='+str(a_list[m][1][0])+';')
    if domain_dim == 1:
        lines.append('hold on')
        for m in range(len(k_list)):
            lines.append('plot(x_'+str(m)+',x_'+str(m)+'.^2,"." )')
            lines.append('plot(x_' + str(m)+',x_' + str(m) +
                         ' .*a_'+str(m) + '+b_'+str(m) + ')')
        lines.append('hold off')
    elif domain_dim == 2:
        lines.append('plot3(...')
        for m in range(len(k_list)-1):
            lines.append('x_'+str(m)+'(:,1),x_'+str(m) +
                         '(:,2),y_'+str(m)+',...')  # \'.\',
        m = len(k_list)-1
        lines.append('x_'+str(m)+'(:,1),x_'+str(m)+'(:,2),y_'+str(m)+')')

    with open('matlab_test.m', 'w') as f:
        f.write('\n'.join(lines))

def iterate_through_generations(generations_num):
    for num_of_generations in range(generations_num):
        results = []
        fun_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
        print(f"{num_of_generations+1} generation: ")
        for n in range(len(fun_list)):
            fun = fun_list[n]
            domain_dim = 10
            upp_bound =10
            ea1 = Evolution(fun, domain_dim, upp_bound, 100, 100, False, 0.1, 0.8)
            ea1.learn()
            points = ea1.get_points_for_approximator(num_of_generations+1, True)
            if n ==1:
                print(points.size)
            approx = Approximator(UPPER_BOUND=upp_bound, DIMENTIONALITY=domain_dim, limit_err=0.01)
            approx.K_list.append(points)
            a = approx.calculate_a_vector(points)
            approx.a_list.append(a)
            a_list, k_list = approx.check_domain()
            results.append([approx.find_minimum(function=fun)[0], approx.find_minimum_in_init_data()[0]])
        for element in results:
            print(element)

def main():
    iterate_through_generations(10)

if __name__ == '__main__':
    main()

#matlab_test(k_list, a_list, domain_dim)