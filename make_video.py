import matplotlib.pyplot as plt
from matplotlib import animation

from src.engine import Node
from src.tetris import TetrisEnv
from src.utils import *

def run_simulation(env, expert_config):
    env.reset()
    state = Node(env)
    ends = False

    while not ends:
        state.prop()
        state, actions = state.expert_actions(expert_config)

        for acts in actions:
            board, reward, ends, _ = env.step(acts)
            yield board

def init_animation():
    return [ax.imshow(np.zeros((20, 10)))]

def animate(i):
    return [ax.imshow(i.T)]


if __name__ == '__main__':
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes()
    plt.axis('off')

    env = TetrisEnv()
    expert_config = {
        't_norm': t_min,
        's_norm': s_max
    }

    anim = animation.FuncAnimation(
        fig, animate,
        lambda: run_simulation(env, expert_config)
    )

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=8, metadata=dict(artist='santiago'))

    anim.save('output/example_max.mp4', writer=writer)
