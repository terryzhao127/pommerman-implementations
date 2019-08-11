from __future__ import annotations
from zero_agent import ZeroAgent

import numpy as np
import pommerman
import settings


class SelfPlay:
    def __init__(self, best_net, num_games, num_simulations, num_exploration_steps):
        """
        Initialize the self-play component
        :param best_net: Current best policy-value network
        :param num_games: Number of games played in each iteration
        :param num_simulations: Number of MCTS simulations to select each action
        """
        self._best_net = best_net
        self._num_games = num_games
        self._zero_agent_1 = ZeroAgent(best_net, num_simulations, is_self_play=True,
                                       num_exploration_steps=num_exploration_steps)
        self._zero_agent_2 = ZeroAgent(best_net, num_simulations, is_self_play=True,
                                       num_exploration_steps=num_exploration_steps)
        self._env = pommerman.make(
            settings.game_config_id,
            [self._zero_agent_1, self._zero_agent_2]
        )

    def start(self):
        """Start self-play and generate data"""
        # TODO: Formalize logs
        training_states = []
        training_prs = []
        training_rewards = []
        for i in range(self._num_games):

            state = self._env.reset()
            done = False
            final_reward = None
            while not done:
                # print('[Self Play] Step %d' % self._env._step_count)

                actions = self._env.act(state)
                state, reward, done, info = self._env.step([action.value for action in actions])
                final_reward = reward

            data_1 = self._zero_agent_1.get_training_data()
            data_2 = self._zero_agent_2.get_training_data()

            training_states += data_1[0][0] + data_2[1][0] + data_1[1][0] + data_2[0][0]
            prs = data_1[0][1] + data_2[1][1] + data_1[1][1] + data_2[0][1]
            training_prs += prs

            if final_reward[0] == 0 and final_reward[1] == 0:
                final_reward[0] = -1
                final_reward[1] = -1
            training_rewards += [final_reward[0]] * (len(prs) // 2) + [final_reward[1]] * (len(prs) // 2)

        return list(zip(training_states, training_prs, training_rewards))[:]
