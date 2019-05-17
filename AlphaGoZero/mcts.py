from __future__ import annotations
from typing import Dict, Tuple, List, NewType, Type
from collections.abc import Hashable
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import time


# TODO:update docs


class MCTS:
    _default_c_puct = 5

    def __init__(
            self,
            init_state: State,
            temp: float,
            time_limit: float = None,
            iteration_limit: int = None,
            c_puct: float = _default_c_puct
    ) -> None:
        if time_limit is not None:
            if iteration_limit is not None:
                raise DuplicateLimitError

            # Time taken for each MCTS search in milliseconds
            if time_limit <= 0:
                raise InvalidLimitError("Time limit must be positive.")

            self._time_limit: float = time_limit
            self._limit_type: _LimitType = _LimitType.TIME
        else:
            if iteration_limit is None:
                raise NoLimitError

            # Number of iterations of the search
            if iteration_limit < 1:
                raise InvalidLimitError("Iteration limit must be greater than one")
            if not isinstance(iteration_limit, int):
                raise InvalidLimitError("Iteration limit must be a integer")

            self._iteration_limit: int = iteration_limit
            self._limit_type: _LimitType = _LimitType.ITERATION

        # noinspection PyTypeChecker
        self._root: _Node = None
        self._temp: float = temp
        self._init_state: State = init_state
        self._c_puct: float = c_puct

    def search(self) -> List[Tuple[List[State.ActionType], Type[np.ndarray]]]:
        # noinspection PyTypeChecker
        self._root: _Node = _Node(self._init_state, None, (1., 1.))

        if self._limit_type == _LimitType.TIME:
            time_limit = time.time() + self._time_limit / 1000
            while time.time() < time_limit:
                self._execute()
        else:
            for i in range(self._iteration_limit):
                self._execute()

        return MCTS._get_pr_tuples(self._root, self._temp)

    def _execute(self) -> None:
        node = self._root
        while not node.state.is_terminal():
            if node.is_expanded:
                node = self._get_best_child_node(node, self._c_puct)
            else:
                action_prs, values = node.state.policy_value()
                self._expand(node, action_prs)
                self._backup(node, values)
                return
        # Terminal state
        values = node.state.get_rewards()
        self._backup(node, values)

    @staticmethod
    def _backup(node: _Node, rewards: State.RewardTuple) -> None:
        while node is not None:
            for i, reward in enumerate(rewards):
                record = node.records[i]

                record.num_visits += 1
                record.q += 1.0 * (reward - record.q) / record.num_visits
                record.w += reward
            node = node.parent

    @staticmethod
    def _expand(node: _Node, action_prs: List[Tuple[State.ActionTuple, Tuple[float]]]) -> None:
        for action_tuple, prob_tuple in action_prs:
            if action_tuple not in node.children.keys():
                new_node = _Node(node.state.take_actions(action_tuple), node, prob_tuple)
                node.children[action_tuple] = new_node
        node.is_expanded = True

    @staticmethod
    def _get_pr_tuples(
            node: _Node,
            temp: float
    ) -> List[Tuple[List[State.ActionType], Type[np.ndarray]]]:
        result = []
        for player_index in range(2):
            actions = node.state.get_possible_actions_for_single_player(player_index)
            action_num_visits: Dict[State.ActionType, int] = {}
            for action in actions:
                num_visits = 0
                for action_tuple, child in node.children.items():
                    if action_tuple[player_index] == action:
                        num_visits += child.records[player_index].num_visits
                action_num_visits[action] = num_visits
            # action_values = MCTS._get_action_values(node, c_puct, player_index)
            # actions = node.state.get_possible_actions_for_single_player(player_index)
            result_num_visits = \
                [x[1] for x in sorted(action_num_visits.items(), key=lambda x: x[0].value)]
            result.append((actions, MCTS._softmax(1.0 / temp * np.log(np.array(result_num_visits) + 1e-10))))

        return result

    @staticmethod
    def _softmax(x) -> Type[np.ndarray]:
        prs = np.exp(x - np.max(x))
        prs /= np.sum(prs)
        return prs

    @staticmethod
    def _get_action_values(
            node: _Node,
            c_puct: float,
            player_index: int
    ) -> Dict[State.ActionType, float]:
        action_values: Dict[State.ActionType, float] = {}
        for action in node.state.get_possible_actions_for_single_player(player_index):
            sum_values = 0
            sum_weights = 0
            for action_tuple, child in node.children.items():
                if action_tuple[player_index] == action:
                    node_value = MCTS._calc_node_value_for_single_player(node, child, c_puct,
                                                                         player_index)
                    weight = child.records[player_index].num_visits + 1  # Prevent zero
                    sum_values += node_value * weight
                    sum_weights += weight
            action_values[action] = sum_values / sum_weights
        return action_values

    @staticmethod
    def _get_best_action_for_single_player(
            node: _Node,
            c_puct: float,
            player_index: int
    ) -> State.ActionType:
        action_values = MCTS._get_action_values(node, c_puct, player_index)
        return max(action_values.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _get_best_child_node(node: _Node, c_puct: float) -> _Node:
        best_actions: List[State.ActionType] = []

        for i in range(node.state.num_players):
            best_actions.append(MCTS._get_best_action_for_single_player(node, c_puct, i))

        for action_tuple, node in node.children.items():
            if action_tuple == tuple(best_actions):
                return node

    @staticmethod
    def _calc_node_value_for_single_player(
            node: _Node,
            child: _Node,
            c_puct: float,
            player_index: int
    ):
        parent_record = node.records[player_index]
        child_record = child.records[player_index]

        child_record.u = c_puct * child_record.p * np.sqrt(parent_record.num_visits) / (1 + child_record.num_visits)
        return child_record.q + child_record.u

    @staticmethod
    def _get_action_tuple(parent, child):
        for action_tuple, node in parent.children.items():
            if node is child:
                return action_tuple


class State(ABC):
    RewardType = None
    ActionType = None
    RewardTuple = NewType('RewardTuple', Tuple[RewardType, ...])
    ActionTuple = NewType('ActionTuple', Tuple[ActionType, ...])

    def __init__(self, player_index, num_players, reward_type, action_type):
        State.RewardType = reward_type
        if not isinstance(action_type, Hashable):
            raise TypeError('The action_type must be hashable')
        State.ActionType = action_type

        self.player_index = player_index
        self.num_players = num_players

    @abstractmethod
    def get_possible_actions_for_single_player(self, player_index: int) -> List[ActionType]:
        """
        Get possible actions for a single player to act

        :return: A list of possible actions
        """
        pass

    @abstractmethod
    def get_possible_actions(self) -> List[ActionTuple]:
        """
        Get possible actions for players to act

        :return: A list of tuples of players' actions
        """
        pass

    @abstractmethod
    def take_actions(self, actions: ActionTuple) -> State:
        """
        Take actions of players and return a new state

        :param actions: A tuple of actions of players
        :return: A new game state
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check whether the current state is the terminal state

        :return: A bool value
        """
        pass

    @abstractmethod
    def get_rewards(self) -> RewardTuple:
        """
        Get rewards of current state for players

        :return: A tuple of rewards for players
        """
        pass

    @abstractmethod
    def policy_value(self) -> Tuple[List[Tuple[ActionTuple, Tuple[float]]], RewardTuple]:
        """
        Get probabilities of actions and value of current state for players

        :return: A list of (action_tuple, pr_tuple) tuples for players
                and a list of scores of the state for each player
        """
        pass


class _Node:
    def __init__(self, state: State, parent: _Node, prior_prs: Tuple[float]):
        self.state: State = state
        self.parent: _Node = parent
        self.is_expanded: bool = False
        self.children: Dict[State.ActionTuple, _Node] = {}

        # Records of players
        self.records: List[_Record] = [_Record(p) for p in prior_prs]


class _Record:
    def __init__(self, prior_pr):
        self.q = 0
        self.u = 0
        self.w = 0
        self.p = prior_pr
        self.num_visits = 0


class _LimitType(Enum):
    TIME = 0
    ITERATION = 1


class LimitError(ValueError):
    pass


class DuplicateLimitError(LimitError):
    def __init__(self):
        super(DuplicateLimitError, self).__init__("Cannot have both a time limit and an iteration limit")


class NoLimitError(LimitError):
    def __init__(self):
        super(ValueError, self).__init__("Must have either a time limit or an iteration limit")


class InvalidLimitError(LimitError):
    def __init__(self, message):
        super(ValueError, self).__init__(message)
