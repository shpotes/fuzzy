from copy import copy, deepcopy
import time
from typing import Dict, Set, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
from src import brain
from src import tetris
from src import utils


class Node:
    """
    Tetris enviroment wrapper, help space explorition
    """
    def __init__(self, env: gym.Env, hist: Dict[int, int] = {}):
        self.env = env
        self.curr_tetrimino = env.tetrimino
        self.hist = hist
        self.childrens = []
        self.fuzzy_var = {}
        self._action_map = [
            'left', 'right',
            'clockwise', 'counter-clockwise',
            'nothing', 'hard-drop'
        ]

    def _get_available_translation(self) -> Set[int]:
        """
        Return all possible translation for a given state
        """
        def cant_go_left(tetrimino: tetris.Tetrimino):
            for x, y in tetrimino.indices:
                if x == 0:
                    return True
            return False

        def cant_go_right(tetrimino: tetris.Tetrimino):
            for x, y in tetrimino.indices:
                if x == 9:
                    return True
            return False

        available_actions = {0, 1, 5}

        if 0 in self.hist or cant_go_right(self.curr_tetrimino):
            available_actions.remove(1)

        if 1 in self.hist or cant_go_left(self.curr_tetrimino):
            available_actions.remove(0)

        if 5 in self.hist:
            return set({})

        return available_actions

    def board(self, with_tetrimino: bool = True):
        """
        Return the current board
        """
        if with_tetrimino:
            return self.env.get_state().T
        return np.copy(self.env.board).T

    def _roof(self):
        """
        Return the index of the last occuped space, for each column
        """
        board = self.board(with_tetrimino=False)
        roof = (board * np.arange(20, 0, -1).reshape(-1, 1)).max(axis=0)
        return roof

    @property
    def number_of_holes(self):
        """
        Return the current number of holes
        """
        board = np.vstack([np.zeros(10),
                           self.board(with_tetrimino=False)])
        roof = self._roof()

        board = 1 - board[::-1]
        board[roof, np.arange(len(roof))] = -20 + roof + 1
        return board[::-1].sum()

    @property
    def pile_height(self):
        """
        return the current height of the largest pile
        """
        return self.column_height - self._roof().min()

    @property
    def column_height(self):
        """
        return the current hight of each column
        """
        return self._roof().max()

    @property
    def wells_height(self):
        """
        return the height of the largest well
        """
        roof = self._roof() - self._roof().min()
        return np.abs(np.diff(roof))[1:].max()

    def __str__(self):
        acts = ''
        for k, v in self.hist.items():
            if k == 4:
                continue
            acts += f'{self._action_map[k]} x {v}, '
        return acts

    def __hash__(self):
        return hash(self.__str__)

    def __eq__(self, other):
        self.hist == other.hist

    def step(self, action: int):
        if action in self.hist:
            self.hist[action] += 1
        else:
            self.hist[action] = 1

        res = self.env.step(action)
        self.curr_tetrimino = self.env.tetrimino
        return res

    def _make_rotation(self, orientation, times):
        """
        rotate the current tetrimonio a certain number
        of times on a certain orientation
        """

        assert orientation in {'clock', 'counter'}
        rotation = {self}
        curr_env, curr_hist = self.env, self.hist

        for _ in range(times):
            new_child = Node(deepcopy(curr_env), copy(curr_hist))
            new_child.step(2 if orientation == 'clock' else 3)
            curr_env, curr_hist = new_child.env, new_child.hist
            rotation.add(new_child)

        return rotation

    def _do_rotations(self):
        """
        Perform all possible rotations
        """
        def number_of_rotation(tetrimino: tetris.Tetrimino):
            if tetrimino.shape == tetris.SHAPES['T']: return 4
            if tetrimino.shape == tetris.SHAPES['J']: return 4
            if tetrimino.shape == tetris.SHAPES['L']: return 4
            if tetrimino.shape == tetris.SHAPES['Z']: return 2
            if tetrimino.shape == tetris.SHAPES['S']: return 2
            if tetrimino.shape == tetris.SHAPES['I']: return 2
            if tetrimino.shape == tetris.SHAPES['O']: return 1

        num_rot = number_of_rotation(self.curr_tetrimino)

        if num_rot == 4:
            return self._make_rotation('clock', 2) | self._make_rotation('counter', 1)
        if num_rot == 2:
            return self._make_rotation('clock', 1)
        return set({self})

    def show(self):
        """
        Plot the current board
        """
        plt.imshow(self.board())
        plt.show()
    
    def _do_translations(self):
        """
        Return all possile child states that can be generated using
        only translation
        """
        available_actions = self._get_available_translation()
        new_childrens = {}
        for action in available_actions:
            new_child = Node(deepcopy(self.env), copy(self.hist))
            _, _, _, meta = new_child.step(action)
            new_childrens[action] = new_child            

            if meta['droped']:
                break
            else:
                new_child._do_translations()

        self.childrens = new_childrens
        return new_childrens

    def prop(self):
        """
        Generate the next generation of states
        """
        whole_childrens = set()
        self.hist = {}

        for child in self._do_rotations():
            for sub_child in child._canonize_tree(child._do_translations()):
                whole_childrens.add(sub_child)

        self.childrens = whole_childrens
        return whole_childrens

    def _canonize_tree(self, childrens):
        """
        Recursively return all the leaf of a three
        """
        leaf = set({})
        if not childrens:
            leaf.add(self)
        else:
            for child in childrens.values():
                leaf |= child._canonize_tree(child.childrens)
        return leaf

    def expert_actions(self, expert_config={}):
        """
        Choose the best child following the FIS result
        """
        self.prop()

        expert = brain.get_expert_model(**expert_config)
        best_action = min(
            self.childrens,
            key=lambda c: brain.make_inference(
                c.number_of_holes - self.number_of_holes,
                c.pile_height - self.pile_height,
                c.wells_height - self.wells_height,
                expert
            )
        )
        return best_action, utils.flatten_counter(best_action.hist)
