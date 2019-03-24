import pommerman
from pommerman import agents
from mcts_agent import MCTSAgent
from multiprocessing import Process, Manager


def run(match_num, iteration_limit, mcts_process_num, result_list=None, process_id=None, render=False):
    """
    Run the match for MCTS and three simple agents.
    :param iteration_limit: The maximal iteration of MCTS
    :param match_num: The number of matches
    :param mcts_process_num: The number of processes used in MCTS
    :param result_list: A list to record results
    :param process_id: The process ID given when you do multiprocessing
    :param render: Determine whether to render game
    :return: None
    """
    if mcts_process_num == 1:
        mcts_process_num = None
    agent_list = [
        MCTSAgent([agents.SimpleAgent for _ in range(3)], iteration_limit, process_count=mcts_process_num),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
    ]

    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for i_episode in range(match_num):
        state = env.reset()
        done = False
        initial_agents = state[0]['alive']
        survivors = initial_agents
        dead_agents = []
        while not done:
            if render:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            survivors = state[0]['alive']
            for agent in initial_agents:
                if agent not in survivors and agent not in dead_agents:
                    dead_agents.append(agent)

        if process_id:
            print('[Process %d, Episode %d] Dead order: ' % (process_id, i_episode),
                  str(dead_agents), 'Survivors:', survivors)
        else:
            print('[Episode %d] Dead order: ' % i_episode, str(dead_agents), 'Survivors:', survivors)
        if result_list:
            result_list.append((dead_agents, survivors))

    if render:
        env.close()


def multiprocessing_run(process_count, match_num, iteration_limit, mcts_process_num):
    """

    :param process_count: The number of processes
    :param match_num: The number of matches
    :param iteration_limit: The maximal iteration of MCTS
    :param mcts_process_num: The number of processes used in MCTS
    :return: None
    """
    with Manager() as manager:
        shared_list = manager.list()

        processes = [Process(target=run, args=(iteration_limit, match_num, mcts_process_num, shared_list, i)) for i in
                     range(process_count)]

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        # Analysis
        print('\n')
        print('=' * 20, '\n')

        mcts_agent = 10
        count = 0
        win_count = 0
        rank_count = {i: 0 for i in range(1, 5)}
        for result in shared_list:
            count += 1

            dead_order = result[0]
            survivor = result[1]
            if mcts_agent in survivor:
                rank_count[1] += 1
                win_count += 1
            else:
                for i in range(len(dead_order)):
                    if dead_order[i] == mcts_agent:
                        rank_count[4 - i] += 1
        print('Match count:', count)
        print('Win count:', win_count)
        print('Average rank:', sum(k * v for k, v in rank_count.items()) / count)


if __name__ == '__main__':
    # run(match_num=20, iteration_limit=10, mcts_process_num=20, render=True)
    multiprocessing_run(process_count=10, match_num=20, iteration_limit=10, mcts_process_num=10)
