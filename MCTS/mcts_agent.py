import json
import random
from collections import Counter

from multiprocessing import Manager, Process
from pommerman.agents import BaseAgent
from pommerman.constants import Action
from ast import literal_eval
import pommerman
import mcts


class MCTSAgent(BaseAgent):
    def __init__(self, opponent_types, iteration_limit, process_count=None):
        """
        Initialize a MCTSAgent.

        :param opponent_types: Types of opponents. These opponents MUST be able to be simulated.
        :param iteration_limit: The maximal iteration of MCTS
        """
        super(MCTSAgent, self).__init__()
        self._iteration_limit = iteration_limit
        self._opponent_types = opponent_types
        self._process_count = process_count
        self._agent_id = None
        self._agent_num = len(opponent_types) + 1
        self._simulation_env = None
        self._game_type = None

    def act(self, obs, action_space):
        state = self._create_simulation_state(obs)

        if self._process_count:
            # Multiprocessing

            def _mcts_search(_state, _agent_id, _simulation_env, _iteration_limit, _shared_list):
                _env_state = _EnvState(_state, _agent_id, _simulation_env)
                _searcher = mcts.MCTS(_env_state, iteration_limit=_iteration_limit)
                _shared_list.append(_searcher.search())

            def get_most_frequent(l):
                count = Counter(l)
                return count.most_common(1)[0][0]

            with Manager() as manager:
                shared_list = manager.list()
                processes = []

                for _ in range(self._process_count):
                    env_copy = self._make_env_copy()
                    processes.append(Process(
                        target=_mcts_search,
                        args=(state, self._agent_id, env_copy, self._iteration_limit, shared_list)
                    ))

                for p in processes:
                    p.start()

                for p in processes:
                    p.join()

                action = get_most_frequent(shared_list)
        else:
            env_state = _EnvState(state, self._agent_id, self._simulation_env)
            searcher = mcts.MCTS(env_state, iteration_limit=self._iteration_limit)
            action = searcher.search()

        return action

    def init_agent(self, id_, game_type):
        super(MCTSAgent, self).init_agent(id_, game_type)
        self._agent_id = id_
        self._game_type = game_type

        # Initialize simulation environment
        self._simulation_env = pommerman.make(pommerman.REGISTRY[game_type.value], self._generate_agents())
        self._simulation_env.reset()

    def _make_env_copy(self):
        state = self._simulation_env.get_json_info()

        new_env = pommerman.make(pommerman.REGISTRY[self._game_type.value], self._generate_agents())
        new_env._init_game_state = state
        new_env.set_json_info()

        return new_env

    def _generate_agents(self):
        agents = []
        for opponent_type in self._opponent_types:
            agents.append(opponent_type())
        agents.insert(self._agent_id, _SimulatedAgent())
        return agents

    def _create_simulation_state(self, obs):
        # Observations from `obs`
        board_size = obs['board'].shape[0]

        step_count = 0
        board = obs['board'].tolist()
        bomb_life = obs['bomb_life'].tolist()
        bomb_blast_strength = obs['bomb_blast_strength'].tolist()
        agents_obs, bombs_obs, flames_obs = MCTSAgent._find_items(board, bomb_life, bomb_blast_strength)
        agents = []
        bombs = []
        flames = []

        # Agents
        for agent_id in range(self._agent_num):
            agent_value = MCTSAgent._agent_id_to_agent_value(agent_id)
            if agent_value in obs['alive']:
                is_alive = True
            else:
                is_alive = False

            if is_alive:
                position = None
                for agent_obs in agents_obs:
                    if agent_obs[0] == agent_value:
                        position = agent_obs[1]
                position = [position[0], position[1]]
            else:
                # Arbitrary position for dead agent.
                position = [0, 0]

            # TODO: Record agent's ability.
            can_kick = False
            blast_strength = 2
            ammo = 1

            agents.append({
                'agent_id': agent_id,
                'is_alive': is_alive,
                'position': position,
                'ammo': ammo,
                'blast_strength': blast_strength,
                'can_kick': can_kick
            })

        # Bombs
        for bomb_obs in bombs_obs:
            position = bomb_obs[0]

            # TODO: Record `bomber_id`.
            bomber_id = random.choice(agents)['agent_id']
            life = bomb_obs[1]
            blast_strength = bomb_obs[2]

            bombs.append({
                'position': [position[0], position[1]],
                'bomber_id': bomber_id,  # Arbitrary bomber_id
                'life': life,
                'blast_strength': blast_strength,
                'moving_direction': None  # TODO: Record `moving_direction`.
            })

        # Flames
        for flame_obs in flames_obs:
            position = flame_obs
            flames.append({
                'position': [position[0], position[1]],
                'life': 1  # TODO: Check whether it is always 1
            })

        # TODO: Randomizing generation of extra items
        extra_items = []

        state = {
            'board_size': board_size,
            'step_count': step_count,
            'board': board,
            'agents': agents,
            'bombs': bombs,
            'flames': flames,
            'items': extra_items,
            'intended_actions': []  # This key is not considered in `env.set_json_info`.
        }

        for key, value in state.items():
            state[key] = json.dumps(value, cls=pommerman.utility.PommermanJSONEncoder)
        return state

    @staticmethod
    def _agent_id_to_agent_value(agent_id):
        return agent_id + 10

    @staticmethod
    def _find_items(board, bomb_life, bomb_blast_strength):
        board_size = len(board[0])
        agent_values = {10, 11, 12, 13}
        bomb_value = 3
        flame_value = 4

        agents = []
        bombs = []
        flames = []

        for x in range(board_size):
            for y in range(board_size):
                if board[x][y] in agent_values:
                    agents.append((board[x][y], (x, y)))
                if board[x][y] == bomb_value:
                    bombs.append(((x, y), bomb_life[x][y], bomb_blast_strength[x][y]))
                if board[x][y] == flame_value:
                    flames.append((x, y))

        return agents, bombs, flames


class _SimulatedAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


class _SimulatedRandomAgent(BaseAgent):
    def act(self, obs, action_space):
        pass


# pommerman.constants.Action is hashable. So we should only implement mcts.State.

class _EnvState(mcts.State):
    def __init__(self, state, agent_id, simulation_env, reward=None, done=False):
        self._state = state
        self._env = simulation_env
        self._agent_id = agent_id
        self._done = done
        self._reward = reward

    def get_possible_actions(self):
        return [a.value for a in Action]

    def take_action(self, action):
        return self._take_action(action)

    def is_terminal(self):
        return self._done

    def get_reward(self):
        if self._reward is not None:
            return self._reward

    def rollout(self):
        env_state = self
        while not env_state.is_terminal():
            env_state = env_state._take_action(is_rollout=True)
        return env_state.get_reward()

    def _update_env(self):
        self._env._init_game_state = self._state
        self._env.set_json_info()
        self._env._intended_actions = [i for i in literal_eval(self._state['intended_actions'])]

    def _take_action(self, action=None, is_rollout=False):
        self._update_env()

        if is_rollout:
            actions = []
            for agent in self._get_agents():
                if agent['is_alive']:
                    actions.append(random.choice([a.value for a in Action]))
                else:
                    actions.append(0)
        else:
            obs = self._env.get_observations()
            actions = self._env.act(obs)
            actions[self._agent_id] = action

        _, rewards, done, _ = self._env.step(actions)
        reward = rewards[self._agent_id]

        new_state = self._env.get_json_info()

        return _EnvState(new_state, self._agent_id, self._env, reward=reward, done=done)

    def _get_agents(self):
        agents = json.loads(self._state['agents'])
        return agents
