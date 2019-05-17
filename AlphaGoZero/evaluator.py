from __future__ import annotations

from zero_agent import ZeroAgent
import pommerman
import settings
import numpy as np


class Evaluator:
    def __init__(self, best_net, new_net, num_games, num_simulations):
        """
        Initialize the evaluation component
        :param best_net: Network 1
        :param new_net: Network 2
        :param num_games: Number of games played in each evaluation
        :param num_simulations: Number of MCTS simulations to select each move
        """
        self._net_1 = best_net
        self._net_2 = new_net
        self._num_games = num_games
        self._env = pommerman.make(
            settings.game_config_id,
            [
                ZeroAgent(best_net, num_simulations=num_simulations, is_self_play=False),
                ZeroAgent(new_net, num_simulations=num_simulations, is_self_play=False),
            ]
        )

    def start(self):
        """Start evaluation and return win ratios of two players"""
        print('Evaluation starts...')

        win_count = np.zeros(2)
        for i in range(self._num_games):

            state = self._env.reset()
            done = False
            reward = None
            print('[Evaluation] Game %d of evaluation started.' % (i + 1))
            while not done:
                # print('[Evaluation] Step %d' % self._env._step_count)
                actions = self._env.act(state)
                state, reward, done, info = self._env.step(actions)
            if reward[0] == settings.win_reward and reward[1] == settings.lose_reward:
                win_count[0] += 1
            elif reward[1] == settings.win_reward and reward[0] == settings.lose_reward:
                win_count[1] += 1

            print('[Evaluation] Game %d of evaluation completed.' % (i + 1))

        result = win_count / win_count.sum()

        return result[1] - result[0] > 0.55
