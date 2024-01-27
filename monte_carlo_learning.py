import numpy as np
import gym
import time
LOOP_SIZE = 100000
m = 3  # Size of action space


# Applying Monte-Carlo along with epsilon greedy
def get_random_policy_for_all_states():
    positions = [state_representation(x) for x in np.arange(-1.2, 0.7, 0.1)]
    velocities = [state_representation(x) for x in np.arange(-0.7, 0.7, 0.1)]
    policy = {}
    for position in positions:
        policy[position] = {}
        for velocity in velocities:
            policy[position][velocity] = []
            for action in (0, 1, 2):
                if action == 2 and position > 0.0:
                    policy[position][velocity].append(4/5)
                else:
                    policy[position][velocity].append(1/3)

    return policy, (positions, velocities)


def initialized_q(states):
    positions = states[0]
    velocities = states[1]
    q_val = {}
    for position in positions:
        q_val[position] = {}
        for velocity in velocities:
            q_val[position][velocity] = []
            for action in range(3):
                if position > 0.1 and action == 2:
                    q_val[position][velocity].append(1)
                else:
                    q_val[position][velocity].append(0)
    return q_val


def state_representation(state):
    return round(state, 1)


def env_policy(state, q_val):
    action_prob = policy_prob(state, q_val)
    return np.random.choice(range(m), p=action_prob)


def policy_prob(state, q_val, epsilon=.3):
    state = (state_representation(state[0]), state_representation(state[1]))
    action_prob = np.ones(m, dtype=float) * epsilon / m
    best_action = get_max(q_val[state[0]][state[1]])
    action_prob[best_action] += (1.0 - epsilon)
    return action_prob


def policy_evaluation(q_val, env):
    s_a_r_list = []
    s_curr = env.reset()
    while s_curr[0] != 0.6:
        a_curr = env_policy(s_curr, q_val)
        s_next, r, done, _ = env.step(a_curr)
        s_a_r_list.append((s_curr, a_curr, r))
        s_curr = (state_representation(s_next[0]), state_representation(s_next[1]))
        env.render()
    return s_a_r_list


def all_action_equal(q_state):
    if q_state[0] == q_state[1]:
        if q_state[1] == q_state[2]:
            return True
    return False


def get_max(q_state):
    max_i = 0
    if q_state[0] == q_state[1] == q_state[2]:
        return np.random.randint(0, 3)
    for index, val in enumerate(q_state):
        if q_state[index] >= q_state[max_i]:
            max_i = index
    return max_i


def simulate_env_monte_carlo(env):
    _, states = get_random_policy_for_all_states()
    Q = initialized_q(states)
    r = {}
    for i in range(LOOP_SIZE):
        start_episode = time.time()
        g_return = 0
        gamma = 0.05
        s_a_r_list = policy_evaluation(Q, env)

        for entry in s_a_r_list[::-1]:
            state = (entry[0][0], entry[0][1])
            g_return = entry[2] + gamma * g_return
            if (state[0], state[0], entry[1]) in r.keys():
                r[(state[0], state[1], entry[1])].append(g_return)
            else:
                r[(state[0], state[1], entry[1])] = [g_return]
            Q[state[0]][state[1]][entry[1]] = np.mean(r[(state[0], state[1], entry[1])])

        print("POLICY UPDATED. Episode: {}. Elapsed time: {}".format(i, time.time()-start_episode))
    print("GG.")


if __name__ == '__main__':
    environment = gym.make('MountainCar-v0')
    environment.reset()
    simulate_env_monte_carlo(environment)
