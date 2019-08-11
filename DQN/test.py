import pommerman

from pommerman import agents
from dqn_agent import DQNAgent
from configs.q5_train_atari_nature import config
from q1_schedule import LinearExploration, LinearSchedule


def main():
    opponents = [
        agents.SimpleAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
    ]

    _train(opponents, train_from_scratch=True)
    _test(opponents, 100, render=False)


def _train(opponents, train_from_scratch=False, render=False):
    env = pommerman.make('PommeFFACompetition-v0', [])

    # Exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # Learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # Initialize agents.
    dqn_agent = DQNAgent(env, config, exp_schedule, lr_schedule, True, train_from_scratch=train_from_scratch)
    dqn_agent_index = _init_agents(env, exp_schedule, lr_schedule, opponents, dqn_agent)

    t = 1
    while t < config.nsteps_train:
        state = env.reset()

        done = False
        while not done:
            t += 1
            if render:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            if reward[dqn_agent_index] == -1 and not done:
                # Stop the episode when the training agent dies.
                dqn_agent.episode_end(-1)
                done = True

    env.close()


def _test(opponents, match_num=20, render=True):
    env = pommerman.make('PommeFFACompetition-v0', [])

    # Exploration strategy
    exp_schedule = LinearExploration(env, 0, 0, 1)

    # Learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # Initialize agents.
    dqn_agent = DQNAgent(env, config, exp_schedule, lr_schedule, False)
    dqn_agent_index = _init_agents(env, exp_schedule, lr_schedule, opponents, dqn_agent)

    count = 0
    win = 0
    for _ in range(match_num):
        state = env.reset()

        done = False

        while not done:
            if render:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if reward[0] == 1:
                win += 1
                print('win at episode %d' % count)

            if reward[dqn_agent_index] == -1 and not done:
                # Stop the episode when the testing agent dies.
                done = True
        count += 1
    print(win / count)

    env.close()


def _init_agents(env, exp_schedule, lr_schedule, opponents, dqn_agent):
    agent_list = [dqn_agent] + opponents
    index = agent_list.index(dqn_agent)
    for id_, agent in enumerate(agent_list):
        agent.init_agent(id_, env.spec._kwargs['game_type'])  # TODO: Don't use protected member.
    env.set_agents(agent_list)

    return index


if __name__ == '__main__':
    main()
