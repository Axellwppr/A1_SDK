import zmq
import threading
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import sys

class RealTimePlotter(QtCore.QObject):
    data_received = QtCore.pyqtSignal(list)  # Signal to communicate with the main thread

    def __init__(self):
        super().__init__()
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plotting")
        self.plots = []
        self.curves = []
        self.data = []

        # Set up ZeroMQ and the thread to receive data
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind("tcp://*:5555")
        self.thread = threading.Thread(target=self.receive_data)
        self.thread.daemon = True
        self.thread.start()

        # Connect the data_received signal to the update_plots slot
        self.data_received.connect(self.update_plots)

        # Set up a timer to regularly update the plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # Update every 50 milliseconds

    def receive_data(self):
        """Receive data in a separate thread and emit it to the main thread."""
        while True:
            data = self.socket.recv_pyobj()  # Receive data from ZeroMQ
            self.data_received.emit(data)  # Emit the data as a signal to the main thread

    def update_plots(self, data):
        """Update the plots in the main thread based on the received data."""
        n = len(data)
        if n != len(self.plots):
            self.win.clear()
            self.plots = []
            self.curves = []
            self.data = [[] for _ in range(n)]
            for i in range(n):
                p = self.win.addPlot(row=i, col=0)
                
                # 固定Y轴范围 (例如设置为 0 到 1)
                p.setYRange(-3, 3)  # 调整上下界范围
                p.setXRange(0, 500)  # 假设你希望在X轴上也固定范围
                
                # 创建曲线，设置线条颜色和粗细
                c = p.plot(pen=pg.mkPen(width=2))  # 设置线条宽度为2像素
                
                self.plots.append(p)
                self.curves.append(c)
                # 在y=0位置画一条水平线
                hline = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(color='r', width=1))  # 红色，线宽1
                p.addItem(hline)
        # Update the data
        for i in range(n):
            self.data[i].append(data[i])
            if len(self.data[i]) > 500:  # Keep the data length under 500
                self.data[i] = self.data[i][-500:]

    def update(self):
        """Update the curves with the latest data."""
        for i, curve in enumerate(self.curves):
            curve.setData(self.data[i])

    def run(self):
        self.app.exec_()

if __name__ == '__main__':
    plotter = RealTimePlotter()
    plotter.run()
