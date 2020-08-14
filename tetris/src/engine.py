from copy import copy, deepcopy
import time
from typing import Dict, Set, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import src.brain
import src.tetris
import src.utils


class Node:
    def __init__(self, env: gym.Env, hist: Dict[int, int] = {}):
        self.env = env
        self.curr_tetrimino = env.tetrimino
        self.hist = hist
        self.childrens = []
        self.fuzzy_var = {}
        self._action_map = [
            'left', 'right',
            'left-rotate', 'right-rotate',
            'nothing', 'hard-drop'
        ]

    # TODO: _get_available_rotations ->> translation
    def _get_available_translation(self) -> Set[int]:
        def cant_go_left(tetrimino: tetris.Tetrimino):
            for x, _ in tetrimino.indices:
                if x == 0:
                    return True
            return False

        def cant_go_right(tetrimino: tetris.Tetrimino):
            for x, _ in tetrimino.indices:
                if x == 9: # Wouldnt work for different board size
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
        if with_tetrimino:
            return self.env.get_state().T
        return np.copy(self.env.board).T

    def _roof(self):
        board = self.board(with_tetrimino=False)
        return (board * np.arange(board.shape[0])[::-1].reshape(-1, 1)).max(axis=0)

    @property
    def number_of_holes(self):
        board = self.board(with_tetrimino=False)
        roof = self._roof()

        board = 1 - board[::-1]
        board[roof, np.arange(len(roof))] = -20 + roof + 1
        return board[::-1].sum()

    @property
    def pile_height(self):
        return self._roof().max() + 1

    @property
    def well_depth(self):
        roof = self._roof()
        roof = roof - roof.min()
        return roof[roof > 0].min()

    def expert_actions(self):
        self.prop()
        self.childrens = self.canonize_tree()

        expert = brain.get_expert_model()
        best_action = max(
            self.childrens,
            key=lambda c: brain.make_inference(c.holes, c.piles, c.wells, expert)
        )

        return best_action, utils.flatten_counter(self.hist)

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

        return self.env.step(action)

    def prop(self):
        available_actions = self._get_available_translation()
        new_childrens = {}
        for action in available_actions:
            new_child = Node(deepcopy(self.env), copy(self.hist))
            new_child.step(action)
            new_child.prop()
            new_childrens[action] = new_child

        self.childrens = new_childrens
        return new_childrens


    def canonize_tree(self):
        leaf = set({})
        if not self.childrens:
            leaf.add(root)
        else:
            for child in self.childrens.values():
                leaf |= canonize_tree(child)
        return leaf
