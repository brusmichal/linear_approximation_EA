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
    fun_list = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    fun_list = [f5, f6, f7, f8]
    #fun_list = [f10]#, f6, f7, f8, f9, f10]
    for n in range(len(fun_list)):
        print(f'Function number: {n+5}')
        results = []
        fun = fun_list[n]
        domain_dim = 10
        upp_bound =10
        population_size = 50
        ea1 = Evolution(goal_function=fun, function_dimension=domain_dim,upper_bound= upp_bound, 
            population_size=population_size, max_iter=generations_num, 
            with_crossing=False, mutation_strength=0.1, p_crossover=0.8)
        ea1.learn()
        points = ea1.get_points_for_approximator(generations_num, True)
        for gen_num in range(generations_num):
            #print((gen_num+1)*population_size)
            pts = points[0:(gen_num+1)*population_size]  
            #print(pts.size) 
            approx = Approximator(UPPER_BOUND=upp_bound, DIMENTIONALITY=domain_dim, limit_err=1)
            approx.K_list.append(pts)
            a = approx.calculate_a_vector(pts)
            approx.a_list.append(a)
            a_list, k_list = approx.check_domain()
            results.append([approx.find_minimum(function=fun)[0], approx.find_minimum_in_init_data()[0]])
        model_is_better = 0
        for element in results:
            print(element)
            if element[0]<element[1]:
                model_is_better+=1
        print(f'Model was better in {model_is_better/len(results)*100}% attempts')

def main():
    iterate_through_generations(generations_num=40)

if __name__ == '__main__':
    main()

#matlab_test(k_list, a_list, domain_dim)