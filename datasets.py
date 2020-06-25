import math
import torch

class NoiseDataset:
    def __init__(self, dim=2):
        self.dim = dim

    def next_batch(self, batch_size=64):
        return torch.randn(batch_size, self.dim)

class ToyDataset:
    def __init__(self, distr='8Gaussians', scale=2, ratio=0.5):
        self.distr = distr
        self.scale = scale
        self.ratio = ratio

        if self.distr == '8Gaussians' or self.distr == 'Imbal-8Gaussians':
            self.std = 0.02
            self.centers = [
                (1, 0),
                (1. / math.sqrt(2), 1. / math.sqrt(2)),
                (1. / math.sqrt(2), -1. / math.sqrt(2)),
                (-1, 0),
                (0, 1),
                (0, -1),
                (-1. / math.sqrt(2), 1. / math.sqrt(2)),
                (-1. / math.sqrt(2), -1. / math.sqrt(2))
            ]
            self.centeroids = torch.FloatTensor(self.centers) * self.scale / 1.414
            self.std *= self.scale / 1.414

        if self.distr == '25Gaussians':
            self.std = 0.05
            self.centers = []
            for x in range(-2, 3):
                for y in range(-2, 3):
                    self.centers.append((x, y))
            self.centeroids = torch.FloatTensor(self.centers) * self.scale

    def next_batch(self, batch_size=64):
        if self.distr == '8Gaussians':
            num_repeat = int(batch_size/8)
            batch = torch.repeat_interleave(self.centeroids, num_repeat, axis=0)
            num_random = batch_size - num_repeat * 8
            if num_random > 0:
                newrow = self.centeroids[torch.randint(8, size=(num_random,)), :]
                batch = torch.cat((batch, newrow))
            noise = torch.empty((batch_size, 2)).normal_(mean=0.0,std=self.std)
            return batch + noise

        if self.distr == 'Imbal-8Gaussians':
            num_rep_maj = int(batch_size*self.ratio/2)
            assert num_rep_maj > 0
            majority = torch.repeat_interleave(self.centeroids[:2, :], num_rep_maj, axis=0)
            num_rep_min = int((batch_size-num_rep_maj*2)/6)
            assert num_rep_min > 0
            minority = torch.repeat_interleave(self.centeroids[2:, :], num_rep_min, axis=0)
            num_random = batch_size-num_rep_maj*2-num_rep_min*6
            if num_random > 0:
                newrow = self.centeroids[torch.randint(8, size=(num_random,)), :]
                batch = torch.cat((majority, minority, newrow))
            else:
                batch = torch.cat((majority, minority))
            noise = torch.empty((batch_size, 2)).normal_(mean=0.0,std=self.std)
            return batch + noise

        if self.distr == '25Gaussians':
            num_repeat = int(batch_size/25)
            batch = torch.repeat_interleave(self.centeroids, num_repeat, axis=0)
            num_random = batch_size - num_repeat * 25
            if num_random > 0:
                newrow = self.centeroids[torch.randint(25, size=(num_random,)), :]
                batch = torch.cat((batch, newrow))
            noise = torch.empty((batch_size, 2)).normal_(mean=0.0,std=self.std)
            return batch + noise
