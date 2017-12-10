from q_learn import *

result = np.zeros((10,10,10))
i = 0
j = 0
k = 0
for gamma in np.linspace(0.8,0.98,4):
    for alpha_C in np.linspace(10,30,5):
        for N_e in [10]:#np.linspace(10,100,10):
            #result[i,j,k] =
            q_learn(gamma, alpha_C,N_e)
    #         k += 1
    #     j+=1
    # i+=1
