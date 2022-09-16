import torch as th


class RunningStat(object):
    """Keeps track of first and second moments (mean and variance) of a streaming time series.
    Taken from https://github.com/joschu/modular_rl
    Math in http://www.johndcook.com/blog/standard_deviation/
    """

    def __init__(self, shape):
        self._n = 0
        self._M = th.zeros(*shape, dtype=th.float32)
        self._S = th.zeros(*shape, dtype=th.float32)

    def push(self, x):
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.clone()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else th.square(self._M)

    @property
    def std(self):
        return th.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, center=True, scale=True, clip=None):
        assert shape is not None
        if clip is not None:
            assert clip > 0

        self.center = center
        self.scale = scale
        self.clip = clip
        self.rs = RunningStat(shape)

    def __call__(self, x, **kwargs):
        self.update_statistics(x)

        if self.center:
            x = x - self.rs.mean
        if self.scale:
            if self.center:
                x = x / (self.rs.std + 1e-8)
            else:
                # diff = x - self.rs.mean
                # diff = diff / (self.rs.std + 1e-8)
                # x = diff + self.rs.mean
                x = (x - self.rs.mean) / (self.rs.std + 1e-8) + self.rs.mean

        if self.clip:
            x = th.clip(x, -self.clip, self.clip)

        return x

    def update_statistics(self, x):
        self.rs.push(x)
