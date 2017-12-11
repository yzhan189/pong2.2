from State import *

Q = np.genfromtxt ('Q_22.csv', delimiter=",")

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




percent = total_wins/n *100
#print(gamma, alpha_C, N_e)
print(total_wins,n)
print(str(percent)+"%")
