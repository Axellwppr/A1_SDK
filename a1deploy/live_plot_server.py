import matplotlib.pyplot as plt
import numpy as np
import socket
import struct


class LivePlotServer:
    def __init__(self, span: float = 2):
        self.span = span
        self.dt = 0.02
        self.fig = None
        self.axes = None
        self.x = None
        self.ys = None
        self.lines = None
        self.bg = None

    def initialize_plot(self, n: int):
        # Initialize the plot based on the new number of plots
        self.n = n
        self.fig, self.axes = plt.subplots(self.n, 1, squeeze=False)
        self.x = list(range(int(self.span / self.dt)))
        self.ys = [[0] * len(self.x) for _ in range(self.n)]
        self.lines = []

        for i in range(self.n):
            ax = self.axes[i, 0]  # Adjusted for single column subplots
            ln = ax.plot(self.x, self.ys[i], animated=True)[0]
            ax.set_ylim(-1, 1)
            self.lines.append(ln)

        plt.show(block=False)
        plt.pause(0.1)

        # Save the current figure background
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.fig.canvas.blit(self.fig.bbox)

    def update_plot(self, yts: list):
        if self.fig is None or len(yts) != self.n:
            # Reinitialize plot if data length is different
            print(f"Data length changed. Reinitializing plot to {len(yts)} plots.")
            self.initialize_plot(len(yts))

        # Restore the figure background before drawing
        self.fig.canvas.restore_region(self.bg)
        
        # Update each line with the new data
        for ax, ln, y, yt in zip(self.axes[:, 0], self.lines, self.ys, yts):
            y.pop(0)
            y.append(yt)
            ln.set_ydata(y)
            ax.draw_artist(ln)

        # Redraw the canvas with updated data
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def start_server(self, ip="0.0.0.0", port=9999):
        # Create UDP socket for receiving data
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((ip, port))
        print(f"Listening on {ip}:{port}")

        while True:
            data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
            num_floats = len(data) // 4  # each float is 4 bytes
            yts = struct.unpack(f'{num_floats}f', data)
            self.update_plot(list(yts))


if __name__ == "__main__":
    server = LivePlotServer()
    server.start_server()
