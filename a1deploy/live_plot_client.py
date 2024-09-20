import socket
import struct
import time
import math


class LivePlotClient:
    def __init__(self, ip="127.0.0.1", port=9999):
        self.ip = ip
        self.port = port
        self.dt = 0.02

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_data(self, yts: list):
        # Pack the data into a binary format before sending
        message = struct.pack(f'{len(yts)}f', *yts)
        self.sock.sendto(message, (self.ip, self.port))

    def test(self):
        for i in range(400):
            t = i * self.dt
            yts = [math.sin(t), math.cos(t)]
            self.send_data(yts)
            time.sleep(self.dt)


if __name__ == "__main__":
    client = LivePlotClient("127.0.0.1", 9999)
    client.test()
