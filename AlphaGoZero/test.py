from pommerman.agents import PlayerAgent, SimpleAgent, RandomAgent
from multiprocessing import Process, Manager
from zero_agent import ZeroAgent
from optimization import Net
import pommerman


def main():
    run(None, 1, render=True)
    # results = multiprocessing_run(None, 14, 1)
    # return analysis(results)


def run(best_net, num_episodes, result_list=None, process_id=None, render=False):
    best_net = Net(trained_model='best.model')
    agent_list = [
        ZeroAgent(net=best_net, num_simulations=100, is_self_play=False, num_exploration_steps=0),
        SimpleAgent()
    ]

    env = pommerman.make('OneVsOne-v0', agent_list)

    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        initial_agents = state[0]['alive']
        survivors = initial_agents
        dead_agents = []
        while not done:
            if render:
                env.render()
            actions = env.act(state)
            actions[0] = actions[0].value
            state, reward, done, info = env.step(actions)

            survivors = state[0]['alive']
            for agent in initial_agents:
                if agent not in survivors and agent not in dead_agents:
                    dead_agents.append(agent)

        if process_id is not None:
            print('[Process %d, Episode %d] Dead order: ' % (process_id, i_episode),
                  str(dead_agents), 'Survivors:', survivors)
        else:
            print('[Episode %d] Dead order: ' % i_episode, str(dead_agents), 'Survivors:', survivors)

        if result_list is None:
            result_list = []
        result_list.append((dead_agents, survivors))

    env.close()

    return result_list


def multiprocessing_run(best_net, process_count, num_episodes):
    with Manager() as manager:
        shared_list = manager.list()

        processes = [Process(target=run, args=(best_net, num_episodes, shared_list, i))
                     for i in range(process_count)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        return [i for i in shared_list]


def analysis(l):
    print('\n')
    print('=' * 20, '\n')

    agent = 10
    count = 0
    win_count = 0
    for result in l:
        if len(result[1]) == 0:
            continue

        count += 1
        survivor = result[1]
        if agent in survivor:
            win_count += 1
    if count == 0:
        print('All matches draw!')
    else:
        print('Win ratio against SimpleAgent:', win_count / count)
        return win_count / count


if __name__ == '__main__':
    main()