from ast import Return
from asyncio.windows_events import NULL
import numpy as np
from numpy.linalg import inv
from numpy import transpose, matmul, subtract, true_divide
from cec2017.functions import *
import sys


class Approximator():
    def __init__(self, UPPER_BOUND, DIMENTIONALITY, limit_err):
        self.UPP_BOUND = UPPER_BOUND  # zakres losowanych punktów
        self.DIM = DIMENTIONALITY  # wymiar dziedziny
        # lista ograniczeń typu >= dla każdego z n wymiarów
        self.up_limits_list = [self.DIM*[self.UPP_BOUND]]
        # lista ograniczeń typu < dla każdego z n wymiarów
        self.low_limits_list = [self.DIM*[-self.UPP_BOUND]]
        # lista parametrów aproksymacji liniowej dla każdej z wydzielonych części dziedziny
        self.a_list = []
        self.K_list = []  # macierz połączona [[1, x, f(x)],...]
        self.lim_err = limit_err  # maksymalny dopuszczalny błąd Y-Y_mod
        self.axis = 1  # oś względem której nastęuje podział dziedziny

    def rnd_x(self):
        #wylosuj punkt x:
        return np.random.uniform(-self.UPP_BOUND, self.UPP_BOUND, size=self.DIM)

    def random_np_array(self, size, function):
        K = np.zeros([size, self.DIM+2], dtype=float)
        #Y = np.zeros([size], dtype=float)
        for n in range(size):
            x = self.rnd_x()
            K[n] = np.append(np.append([1], x), function(x))
        return K

    def calculate_a_vector(self, K):
        Y = K[:, -1:]
        M = K[:, :-1]
        MtM_inv = inv(M.T @ M)
        a = MtM_inv @ M.T @ Y
        return a

    #wartość modelu dla podanego w K wektowa argumentów
    def get_model_output(self, K, a):
        M = K[:, :-1]
        Y_mod = M @ a
        return Y_mod

    def get_sq_err(self, Y, Y_mod):
        err_sq = (Y - Y_mod).T  @ (Y - Y_mod)
        return err_sq

    #znajdowanie indeksu wyjścia modelu, gdzie indeks argumentu będzie identyczny, któremu odpowiada największy błąd Y-Y_mod
    def find_max_err(self, Y, Y_mod):
        max_err = 0
        n_max = 0
        if len(Y) > 2*(self.DIM+1):
            for n in range(self.DIM+1, (len(Y)-self.DIM-1)):
                err = (Y[n] - Y_mod[n])**2
                #print(f'Err: {err}')
                if err > max_err:
                    max_err = err
                    n_max = n
            #print(f"Biggest error was found at {n_max} of {len(Y)} elements")
        else:
            n_max = -1
        return n_max, max_err

    #zmiana osi wzdłóż której nastęuje podział częsci dziedziny
    def change_axis(self):
        if self.DIM == 1:
            self.axis = 1
        if self.axis == self.DIM:
            self.axis = 1
        else:
            self.axis += 1

    #sprawdzenie danej części dziedziny pod kątem największego błędu, dla argumentów wg. których
    #możliwy jest podział dziedziny
    def check_domain_part(self, K, a):
        Y = K[:, -1:]
        M = K[:, :-1]
        #K = K[K[:, self.axis+1].argsort()]
        Y_mod = self.get_model_output(K, a)
        n, max_err = self.find_max_err(Y, Y_mod)
        return n, max_err

    #podział dziedziny w podanym miejscu i wygenerowanie nowego modelu
    def split_domain(self, K, n):
        # print(f'Max err: {max_err}, n: {n}')
        A = K[:n, :]
        B = K[n:, :]
        a_A = self.calculate_a_vector(A)
        a_B = self.calculate_a_vector(B)
        split_value = K[n, self.axis]
        return A, B, a_A, a_B, split_value


    def check_domain(self, K_not_included=[]):
        i = 0
        m_max, err_max = self.check_all_parts(K_not_included)
        while err_max > self.lim_err:
            if len(K_not_included) == len(self.K_list):
                print('Impossible to approximate')
                return self.a_list, self.K_list
            elif i == self.DIM:
                print("To few points to split domain part.")
                K_not_included.append(m_max)
                print(f"Excluding {m_max} part")
                m_max, err_max = self.check_all_parts(K_not_included)
                if m_max == -1:
                    break
                i = 0
            else:
                #print(f'Axis: {self.axis}')
                curr_K = self.K_list[m_max][self.K_list[m_max][:, self.axis].argsort()]
                n, err = self.check_domain_part(curr_K, self.a_list[m_max])
                if n == -1:
                    print(
                        f"To few points to split for {m_max} part of {len(self.K_list)}")
                    i = self.DIM
                else:
                    split_value = self.update_K_a_b_lists(curr_K, n, m_max)
                    self.manage_bounds_set(m_max, split_value)
                    #print("Split of domain done")
                    m_max, err_max = self.check_all_parts(K_not_included)
                    if m_max == -1:
                        break
                self.change_axis()
        return self.a_list, self.K_list

    def update_K_a_b_lists(self, curr_K, n, m_max):
        A, B, a_A, a_B, split_value = self.split_domain(curr_K, n)
        self.K_list[m_max] = A
        self.K_list.append(B)
        self.a_list[m_max] = a_A
        self.a_list.append(a_B)
        return split_value

    def manage_bounds_set(self, m_max, value):
        length = len(self.up_limits_list)
        self.up_limits_list.append(list(self.up_limits_list[m_max]))
        self.up_limits_list[m_max][self.axis-1] = value
        self.low_limits_list.append(list(self.low_limits_list[m_max]))
        self.low_limits_list[len(self.up_limits_list)-1][self.axis-1] = value

    def check_all_parts(self, K_not_included):
        m_max = -1
        err_max = 10000
        #print(f"Checking everything. Not including {len(K_not_included)} parts.")
        for m in range(len(self.K_list)):
            if m in K_not_included:
                continue
            n, err = self.check_domain_part(self.K_list[m], self.a_list[m])
            if err > err_max:
                err_max = err
                m_max = m
            #print(f"Not excluded: {m}")
        return m_max, err_max

    def find_minimum(self, function):
        min = 10e10
        x_min = NULL
        for m in range(len(self.K_list)):
            low_bound = list(self.low_limits_list[m])
            upp_bound = list(self.up_limits_list[m])
            a = list(self.a_list[m])
            #number of coordinates of an x which reach to the edge of the domain
            dim_at_max = 0
            for dim in range(self.DIM):
                if upp_bound[dim] == self.UPP_BOUND:
                    x = list(low_bound)
                    x[dim] = upp_bound[dim]
                    min, x_min = self.minimum_test(a, min,x_min, x, function)
                    dim_at_max +=1
            min, x_min = self.minimum_test(a, min,x_min, list(low_bound), function)
            if dim_at_max == self.DIM:
                min, x_min = self.minimum_test(a, min,x_min, list(upp_bound), function)
        return min, x_min

    def minimum_test(self, a, min, x_min, x, function):
        x_norm = np.append([1], x)
        value = x_norm @ a
        value = function(np.array(x))
        if value < min:
            min = value
            x_min = x
        return min, x_min





def x_square(x):
    sum = 0
    for n in range(x.shape[0]):
        sum += x[n]**2
    return sum


domain_dim = 2
approx = Approximator(UPPER_BOUND=100, DIMENTIONALITY=domain_dim, limit_err=0.01)
K = approx.random_np_array(size=10000, function=x_square)
approx.K_list.append(K)
a = approx.calculate_a_vector(K)
approx.a_list.append(a)
a_list, k_list = approx.check_domain()
print(approx.find_minimum(function=x_square))


def matlab_test():
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

matlab_test()
