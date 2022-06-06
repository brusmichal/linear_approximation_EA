from ast import Return
import numpy as np
from numpy.linalg import inv
from numpy import transpose, matmul, subtract
from cec2017.functions import *
import sys



class Approximator():
    def __init__(self, UPPER_BOUND, DIMENTIONALITY, limit_err):
        self.UPP_BOUND = UPPER_BOUND #zakres losowanych punktów
        self.DIM = DIMENTIONALITY #wymiar dziedziny
        self.up_limits_list = [] #lista ograniczeń typu >= dla każdego z n wymiarów
        self.low_limits_list = [] #lista ograniczeń typu < dla każdego z n wymiarów
        self.a_list = [] #lista parametrów aproksymacji liniowej dla każdej z wydzielonych części dziedziny
        self.K_list = [] #macierz połączona [[1, x, f(x)],...]
        self.lim_err = limit_err #maksymalny dopuszczalny błąd Y-Y_mod
        self.axis = 1 #oś względem której nastęuje podział dziedziny


    def rnd_x(self):
        #wylosuj punkt x:
        return np.random.uniform(-self.UPP_BOUND, self.UPP_BOUND, size=self.DIM)

    def random_np_array(self,size, function):
        K = np.zeros([size,self.DIM+2], dtype=float)
        #Y = np.zeros([size], dtype=float)
        for n in range(size):
            x = self.rnd_x()
            K[n] = np.append(np.append([1],x),function(x))
        self.K_list.append(K)
        return K

    def calculate_a_vector(self,K):
        Y = K[:,-1:]
        M = K[:,:-1]
        MtM_inv = inv(M.T @ M)
        a = MtM_inv @ M.T @ Y
        return a

    #wartość modelu dla podanego w K wektowa argumentów
    def get_model_output(self,K,a):
        M = K[:,:-1]
        Y_mod =  M @ a
        return Y_mod

    def get_sq_err(self,Y,Y_mod):
        err_sq = (Y - Y_mod).T  @ (Y - Y_mod)
        return err_sq

    #znajdowanie indeksu wyjścia modelu, gdzie indeks argumentu będzie identyczny, któremu odpowiada największy błąd Y-Y_mod
    def find_max_err(self,Y,Y_mod):
        max_err = 0
        n_max = 0
        if len(Y) >=2*(self.DIM+1):
            for n in range(self.DIM+1, (len(Y)-self.DIM-1)):
                
                err = (Y[n] - Y_mod[n])**2
                print(f'Err: {err}')
                if err>max_err:
                    max_err = err
                    n_max = n
                    print(n_max)
        else:
            n_max = -1
        return n_max,max_err
    
    #zmiana osi wzdłóż której nastęuje podział częsci dziedziny
    def change_axis(self):
        if self.DIM ==1:
            return
        if self.axis == self.DIM:
            self.axis = 1
        else:
            self.axis += 1

    #sprawdzenie danej części dziedziny pod kątem największego błędu, dla argumentów wg. których
    #możliwy jest podział dziedziny
    def check_domain_part(self,K,a):
        Y = K[:,-1:]
        M = K[:,:-1]
        #K = K[K[:, self.axis+1].argsort()]
        Y_mod = self.get_model_output(K,a)
        n,max_err = self.find_max_err(Y,Y_mod)
        return n, max_err

    #podział dziedziny w podanym miejscu i wygenerowanie nowego modelu
    def split_domain(self, K,n):
        # print(f'Max err: {max_err}, n: {n}')
        A = K[:n,:]
        B = K[n:,:]
        a_A = self.calculate_a_vector(A)
        a_B = self.calculate_a_vector(B)
        return A,B,a_A,a_B

    def check_domain(self, K_not_included = []):
        K_not_included = []
        err_max = 0
        i=0
        m_max, err_max = self.check_all_parts(K_not_included)
        while err_max>self.lim_err :#(n < self.DIM+1 or n> (self.K_list[m_max].shape[0]-self.DIM-1)):
            if len(K_not_included)==len(self.K_list):
                print('Impossible to approximate')
                return self.a_list, self.K_list
            elif i == self.DIM+1:
                print("To few points to split domain part.")
                K_not_included.append(m_max)
                m_max, err_max = self.check_all_parts(K_not_included)
            else:
                print(f'Axis: {self.axis}')
                curr_K = self.K_list[m_max][self.K_list[m_max][:, self.axis].argsort()]
                n, err = self.check_domain_part(curr_K,self.a_list[m_max])
                if n==-1:
                    i+=1
                else:
                    A,B,a_A,a_B = self.split_domain(curr_K,n)
                    self.K_list[m_max] = A
                    self.K_list.append(B)
                    self.a_list[m_max] = a_A
                    self.a_list.append(a_B)
                    i=0
                self.change_axis()
        return self.a_list, self.K_list


    def check_all_parts(self, K_not_included):
        m_max = 0
        err_max = 0
        for m in range(len(self.K_list)):
            if m in K_not_included:
                continue
            n,err = self.check_domain_part(self.K_list[m], self.a_list[m])
            if err>err_max:
                err_max = err
                m_max = m
        return m_max, err_max




def x_square(x):
    sum = 0
    for n in range(x.shape[0]):
        sum += x[n]**2
    return sum

approx = Approximator(UPPER_BOUND=10,DIMENTIONALITY=1, limit_err=0.1)
K = approx.random_np_array(size = 50,function = x_square)
# Y = K[:,-1:]
np.set_printoptions(threshold=np.inf)
a=approx.calculate_a_vector(K)
approx.a_list.append(a)
print(-np.inf)
a_list, k_list = approx.check_domain()
print('All a vect:')
print(a_list)
print('All K vect:')
print(k_list)

for m in range(len(k_list)):
    abc = 'abcdefghijklmnopqrstuvwxyz'
    print('x_'+abc[m]+'= [' )
    for n in range(k_list[m].shape[0]):
        print(k_list[m][n][1:k_list[m].shape[1]-1][0])
    print(']')
    print('b_'+abc[m]+'='+str(a_list[m][0][0]) )
    print('a_'+abc[m]+'='+str(a_list[m][1][0]) )
print('hold on')
for m in range(len(k_list)):
    print('plot(x_'+abc[m]+',x_'+abc[m]+'.^2,"." )')
    print('plot(x_' +abc[m]+',x_' +abc[m]+' .*a_'+abc[m]+ '+b_'+abc[m]+ ')')
print('hold off')