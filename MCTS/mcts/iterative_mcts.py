from . import MCTS
from .base_mcts import _Node
import time
from six.moves import queue


class IterativeMCTS(MCTS):
    def search(self):
        if self._root is None:
            self._root = _Node(self._initial_state, None)

        if self._limit_type == 'time':
            time_limit = time.time() + self.time_limit / 1000
            while time.time() < time_limit:
                self._execute()
        else:
            for i in range(self._search_limit):
                self._execute()

        best_child = MCTS._get_best_child(self._root, 0)
        action = MCTS._get_action(self._root, best_child)

        return action

    def update(self, action):
        old_root = self._root
        for a, node in self._root.children.items():
            if a == action:
                node.parent = None
                self._root = node
                break

        if self._root == old_root:
            self._root = _Node(self._root.state.take_action(action), None)

        old_root.children.pop(action, None)
        IterativeMCTS._destroy_tree(old_root)

    # Following two functions are a NAIVE solutions to delete useless nodes restricted to Python's GC mechanisms.

    @staticmethod
    def _destroy_tree(root):
        q = queue.Queue()
        q.put(root)
        is_root = True

        while not q.empty():
            node = q.get()
            for child_node in node.children.values():
                q.put(child_node)

            # If reference of root is deleted, the outer (caller of class) reference of state will be deleted too.
            if not is_root:
                IterativeMCTS._destroy_object(node.state)
            IterativeMCTS._destroy_object(node)

            is_root = False

    @staticmethod
    def _destroy_object(o):
        members = [attr for attr in dir(o) if not callable(getattr(o, attr)) and not attr.startswith('__')]
        for member in members:
            delattr(o, member)
        del o
