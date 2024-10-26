# From https://github.com/luizgh/visdom_logger
import socket

import visdom
import torch
from collections import defaultdict


class VisdomLogger:
    def __init__(self, port=8097, env="main", server="localhost"):
        self.vis = visdom.Visdom(port=port, env=env, server=server)
        self.windows = defaultdict(lambda: None)
        self.port = port
        self.env = env

    def scalar(self, name, x, y):
        win = self.windows[name]

        update = None if win is None else 'append'
        win = self.vis.line(torch.Tensor([y]), torch.Tensor([x]),
                            win=win, update=update, opts={'legend': [name]})

        self.windows[name] = win

    def scalars(self, list_of_names, x, list_of_ys):
        name = '$'.join(list_of_names)

        win = self.windows[name]

        update = None if win is None else 'append'
        list_of_xs = [x] * len(list_of_ys)
        win = self.vis.line(torch.Tensor([list_of_ys]), torch.Tensor([list_of_xs]),
                            win=win, update=update, opts={'legend': list_of_names})

        self.windows[name] = win

    def images(self, name, images, mean_std=None):
        win = self.windows[name]

        win = self.vis.images(images if mean_std is None else
                              images * torch.Tensor(mean_std[0]) + torch.Tensor(mean_std[1]),
                              win=win, opts={'legend': [name]})

        self.windows[name] = win

    def reset_windows(self):
        self.windows.clear()

    def save(self, envs=["main"]):
        self.vis.save(envs)

    def get_ip(self):
        st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            st.connect(('10.255.255.255', 1))
            IP = st.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            st.close()
        return IP

    def get_visodom_address(self):
        return "http://{}:{}/env/{}".format(self.get_ip(), self.port, self.env)
