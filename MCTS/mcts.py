import time
import math
import random


class MCTS:
    _default_exploration_constant = 1 / math.sqrt(2)

    def __init__(self, time_limit=None, iteration_limit=None, exploration_constant=_default_exploration_constant,
                 rollout_policy=None, two_players=False):
        if time_limit is not None:
            if iteration_limit is not None:
                raise _DuplicateLimitError

            # time taken for each MCTS search in milliseconds
            if time_limit <= 0:
                raise _InvalidLimitError("Time limit must be positive.")

            self.time_limit = time_limit
            self._limit_type = 'time'
        else:
            if iteration_limit is None:
                raise _NoLimitError

            # number of iterations of the search
            if iteration_limit < 1:
                raise _InvalidLimitError("Iteration limit must be greater than one")
            if not isinstance(iteration_limit, int):
                raise _InvalidLimitError("Iteration limit must be a integer")

            self._search_limit = iteration_limit
            self._limit_type = 'iterations'
        self._exploration_constant = exploration_constant
        self._rollout_policy = rollout_policy if rollout_policy is not None else MCTS._random_policy
        self._root = None
        self._two_players = two_players

    def search(self, initial_state):
        self._root = _Node(initial_state, None)

        if self._limit_type == 'time':
            time_limit = time.time() + self.time_limit / 1000
            while time.time() < time_limit:
                self._execute()
        else:
            for i in range(self._search_limit):
                self._execute()

        best_child = MCTS._get_best_child(self._root, 0)
        return MCTS._get_action(self._root, best_child)

    def _execute(self):
        node = self._tree_policy(self._root)
        reward = self._rollout_policy(node)
        self._backpropogate(node, reward)

    def _tree_policy(self, node):
        while not node.is_terminal:
            if node.is_fully_expanded:
                node = MCTS._get_best_child(node, self._exploration_constant)
            else:
                return MCTS._expand(node)
        return node

    @staticmethod
    def _backpropogate(node, reward):
        while node is not None:
            node.num_visits += 1
            node.total_reward += reward
            node = node.parent

            reward = -reward

    @staticmethod
    def _get_best_child(node, exploration_value):
        best_value = float('-inf')
        best_nodes = []
        for child in node.children.values():
            node_value = MCTS._calc_node_value(node, child, exploration_value)

            if node_value > best_value:
                best_value = node_value
                best_nodes = [child]
            elif node_value == best_value:
                best_nodes.append(child)
        return random.choice(best_nodes)

    @staticmethod
    def _calc_node_value(node, child, exploration_value):
        return child.total_reward / child.num_visits + exploration_value * \
               math.sqrt(2 * math.log(node.num_visits) / child.num_visits)

    @staticmethod
    def _get_action(parent, child):
        for action, node in parent.children.items():
            if node is child:
                return action

    @staticmethod
    def _expand(node):
        actions = node.state.get_possible_actions()
        for action in actions:
            if action not in node.children.keys():
                new_node = _Node(node.state.take_action(action), node)
                node.children[action] = new_node
                if len(actions) == len(node.children):
                    node.is_fully_expanded = True
                return new_node

    @staticmethod
    def _random_policy(node):
        state = node.state
        while not state.is_terminal():
            action = random.choice(state.get_possible_actions())
            state = state.take_action(action)
        return state.get_reward() if state.winner() == node.parent.state.current_player else -state.get_reward()


class _Node:
    def __init__(self, state, parent):
        self.state = state
        self.is_terminal = state.is_terminal()
        self.is_fully_expanded = self.is_terminal
        self.children = {}

        # Information for last player to decide
        self.parent = parent
        self.num_visits = 0
        self.total_reward = 0


class _LimitError(ValueError):
    pass


class _DuplicateLimitError(_LimitError):
    def __init__(self):
        super(ValueError, self).__init__("Cannot have both a time limit and an iteration limit")


class _NoLimitError(_LimitError):
    def __init__(self):
        super(ValueError, self).__init__("Must have either a time limit or an iteration limit")


class _InvalidLimitError(_LimitError):
    def __init__(self, message):
        super(ValueError, self).__init__(message)
