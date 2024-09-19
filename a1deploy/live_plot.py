import matplotlib.pyplot as plt
import numpy as np


class LivePlot:
    def __init__(self, n: int, span: float=2):
        self.n = n
        self.span = span
        self.dt = 0.02

        self.fig, self.axes = plt.subplots(n, 1)
        self.x = list(range(int(span / self.dt)))
        self.ys = [[0] * len(self.x) for _ in range(n)]
        self.lines = []
        for ax, y in zip(self.axes, self.ys):
            ln = ax.plot(self.x, y, animated=True)[0] 
            ax.set_ylim(-np.pi, np.pi)
            self.lines.append(ln)
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.fig.canvas.blit(self.fig.bbox)

    def update(self, yts: list):
        self.fig.canvas.restore_region(self.bg)
        for ax, ln, y, yt in zip(self.axes, self.lines, self.ys, yts):
            y.pop(0)
            y.append(yt)
            ln.set_ydata(y)
            ax.draw_artist(ln)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
            

if __name__ == "__main__":
    import time
    import math
    live_plot = LivePlot(2)

    for i in range(400):
        t = i * live_plot.dt
        live_plot.update([math.sin(t), math.cos(t)])
        time.sleep(live_plot.dt)
