import numpy as np
import matplotlib.pyplot as plt
from Enviroment import HopfieldEnvironment

class HopfieldNudgeEnvironment(HopfieldEnvironment):
    def __init__(self, nudge_ammount=3, *args, **kwargs):
        self.nudge_ammount = nudge_ammount
        super().__init__(*args, **kwargs)
    def actions(self):
        return {"index": dict(type='int', num_values=self.size),
                "nudgeX": dict(type='int', num_values=2 * self.nudge_ammount + 1),
                "nudgeY": dict(type='int', num_values=2 * self.nudge_ammount + 1),
                "angle": dict(type=int, num_values=8)}
    
    def do_action(self, action):
        start_pos = self.pos[action["index"]]
        angle = np.pi *2 ** action["angle"] / 180
        end_pos = (start_pos[0] + action["nudgeX"] - self.nudge_ammount,
                   start_pos[1] + action["nudgeY"] - self.nudge_ammount)
        end_pos = tuple(np.clip(end_pos, 0, self.size - 1))
        self.swap(start_pos, end_pos, angle)

    def make_random_action(self):
        index = np.random.randint(self.size)
        nudgeX = np.random.randint(2 * self.nudge_ammount + 1)
        nudgeY = np.random.randint(2 * self.nudge_ammount + 1)
        angle = np.random.randint(8)
        return {"index": index, "nudgeX": nudgeX, "nudgeY": nudgeY, "angle": angle}


if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation
    env = HopfieldNudgeEnvironment(size=100)
    fig, ax = plt.subplots()

    def animate(i):
        plt.cla()
        plt.title(f"Step {i*10}")
        for _ in range(10):
            env.do_action(env.make_random_action())
        env.plot()
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_aspect('equal')
        ax.grid(alpha=0.9, linewidth=0.5)

    ani = FuncAnimation(fig, animate, interval=100, frames=100)
    plt.show()


