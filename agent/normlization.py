# _*_ coding: UTF-8 _*_

import torch


class RunningStatistic:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self._n = 0
        self._mean = torch.zeros(shape, dtype=torch.float)
        self._S = torch.zeros(shape, dtype=torch.float)

    def update_batch(self, batch):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#cite_note-:0-10
        """
        # batch = np.asarray(batch, dtype=np.float32)
        if batch.ndim == 1 and self._mean.ndim == 0:
            batch = batch.reshape(-1, 1)
        elif batch.ndim == 1:
            batch = batch[:, None]

        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0, unbiased=False)

        n_old = self._n
        n_new = self._n + batch_count
        delta = batch_mean - self._mean

        self._mean = self._mean + delta * batch_count / n_new

        self._S = self._S + batch_var * batch_count + delta ** 2 * n_old * batch_count / n_new
        self._n = n_new

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._mean

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else torch.ones_like(self._S)

    @property
    def std(self):
        return torch.sqrt(self.var)

    @property
    def shape(self):
        return self._mean.shape


class Normalization:
    def __init__(self, shape, center=True, scale=True, clip=None):
        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStatistic(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.rs.update_batch(x)

        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                diff = x - self.rs.mean
                diff = diff / (self.rs.std + 1e-8)
                x = diff + self.rs.mean
        if self.clip:
            x = torch.clip(x, -self.clip, self.clip)
        return x


class RewardScaling:
    def __init__(self, shape, gamma, clip=None):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.clip = clip
        self.rs = RunningStatistic(shape=self.shape)
        self.ret = torch.zeros(self.shape, dtype=torch.float)

    def __call__(self, x):
        self.ret = self.gamma * self.ret + x
        self.rs.update_batch(self.ret)
        x = x / (self.rs.std + 1e-8)  # Only divided std
        if self.clip:
            x = torch.clip(x, -self.clip, self.clip)
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.ret = torch.zeros_like(self.ret)
