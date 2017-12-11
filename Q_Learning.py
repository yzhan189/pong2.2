from State import *
import random
import time

REWARD_WIN = 1

def q_learn_and_store(gamma, alpha_C,N_e):
    # discount factor
    gamma = gamma
    # learning rate, start at 1, decay as O(1/t), t starts as 1
    def alpha(t):
        return alpha_C/(alpha_C+t-1)

    def f_function(u_action,n):
        if n < N_e:
            return random.randint(0,2)
        else :
            return u_action

    # Q(s,a) (up,stay,down)
    # Q = np.zeros((124418,3))
    # N = np.zeros((124418,3))
    Q = np.genfromtxt ('Q_22.csv', delimiter=",")
    N = np.genfromtxt ('N_22.csv', delimiter=",")


    n = 0
    t = 0
    diff = 0

    state = State()
    s = state.discretize_get_index()
    a_t = random.randint(0,2)

    start_time = time.time()
    while n<100000:
        # terminal state
        if s >= 124416:
            n +=1
            if s==124416:
                Q[s] = [-1,-1,-1]
            else:
                Q[s] = [1,1,1]
            if n%1000 ==0:
                print(n)
            # start a new trial
            state = State()
            s = state.discretize_get_index()


        else:
            # get action
            N[s,a_t] += 1

            # take action change state
            if a_t == 0:
                state.move_paddle_up()
            elif a_t == 2:
                state.move_paddle_down()

            # reward of taking a_t at s
            r_t = state.move_ball_get_rewards()

            # get next state
            s_prime =  state.discretize_get_index()

            alph = alpha(N[s,a_t])
            a_prime = np.argmax(Q[s_prime])
            Q[s,a_t] = Q[s,a_t] + alph*(r_t+gamma*max(Q[s_prime])-Q[s,a_t])
            #(1-alph) * Q[s,a_t] + alph*(r_t + gamma*np.max(Q[s_prime]))
            if s != s_prime:
                diff += 1

            a_t = f_function(a_prime, N[s_prime,a_prime])
            s = s_prime
            t += 1

    end_time = time.time()
    print(n,t, diff)
    print(end_time-start_time)

    np.savetxt("Q_22.csv", Q, delimiter=",")
    np.savetxt("N_22.csv", N, delimiter=",")

    #Q = np.genfromtxt ('part22.csv', delimiter=",")

    n = 0
    total_wins = 0
    state = State()
    while n<1000:
        index = state.discretize_get_index()
        action = np.argmax(Q[index])
        if action ==0:
            state.move_paddle_up()
        elif action ==2:
            state.move_paddle_down()

        reward = state.move_ball_get_rewards()

        if reward==-1 or reward == REWARD_WIN:
            if reward ==REWARD_WIN:
                total_wins+=1
            state = State()
            n += 1




    ave = total_wins/n
    #print(gamma, alpha_C, N_e)
    print(total_wins,n,ave)
    print()
    return ave


q_learn_and_store(0.9,20,10)
