import gym
import time

from monte_carlo_learning import get_random_policy_for_all_states, initialized_q, env_policy, state_representation

EPISODES = 100


def simulate_env_monte_carlo(env):
    _, states = get_random_policy_for_all_states()
    Q = initialized_q(states)
    max_steps = 5000
    gamma = 0.1
    alpha = 0.01

    for episode in range(EPISODES):
        start_episode = time.time()
        t = 0
        curr_state = env.reset()
        curr_action = env_policy(curr_state, Q)

        while t < max_steps:
            env.render()
            next_state, reward, _, _ = env.step(curr_action)
            next_state = (state_representation(next_state[0]), state_representation(next_state[1]))
            curr_state = (state_representation(curr_state[0]), state_representation(curr_state[1]))
            next_action = env_policy(next_state, Q)

            predicted = Q[curr_state[0]][curr_state[1]][curr_action]
            target = reward + gamma * Q[next_state[0]][next_state[1]][next_action]
            Q[curr_state[0]][curr_state[1]][curr_action] = predicted + alpha * (target - predicted)

            curr_state = next_state
            curr_action = next_action

            t += 1

            if curr_state[0] == 0.6:
                break
        print("End of episode: {}. Elapsed time: {}".format(episode,  time.time()-start_episode))
    print("GG")


if __name__ == '__main__':
    environment = gym.make('MountainCar-v0')
    environment.reset()
    simulate_env_monte_carlo(environment)


