import torch, torchvision

class mnist:
    def __init__(self):
        self.mnist_train = torchvision.datasets.MNIST("/tmp/mnist", download=True, train=True)
        self.mnist_test = torchvision.datasets.MNIST("/tmp/mnist", download=True, train=False)
    @property
    def x_shape(self):
        return [1, 28, 28]
    @property
    def x_min(self):
        return 0.
    @property
    def x_max(self):
        return 1.
    @property
    def target_min(self):
        return 0.
    @property
    def target_max(self):
        return 9.
    def data_preproc(self, x):
        return x[:,None,:,:].float()/255.
    def target_preproc(self, y):
        return y.float()
    def train_sample(self, n):
        idx = torch.randint(0, self.mnist_train.data.shape[0], [n])
        return self.data_preproc(self.mnist_train.data[idx]), self.target_preproc(self.mnist_train.targets[idx])
    def test_sample(self, n):
        current_n = 0
        while current_n + n < self.mnist_test.data.shape[0]:
            idx = torch.arange(current_n, current_n+n)
            yield self.data_preproc(self.mnist_test.data[idx]), self.target_preproc(self.mnist_test.targets[idx].float())
            current_n += n

class mnist_aug(mnist):
    def aug_sample(self, n):
        return torch.rand([n]+self.x_shape), torch.zeros(n)

    def train_sample(self, n):
        idx = torch.randint(0, self.mnist_train.data.shape[0], [n//2])
        mnist_x, mnist_y = self.data_preproc(self.mnist_train.data[idx]), self.target_preproc(self.mnist_train.targets[idx])
        aug_x, aug_y = self.aug_sample(n//2)
        return torch.cat([mnist_x, aug_x], 0), torch.cat([mnist_y, aug_y], 0)

    def test_sample(self, n):
        current_n = 0
        while current_n + n//2 < self.mnist_test.data.shape[0]:
            idx = torch.arange(current_n, current_n+n//2)
            mnist_x, mnist_y = self.data_preproc(self.mnist_test.data[idx]), self.target_preproc(self.mnist_test.targets[idx].float())
            aug_x, aug_y = self.aug_sample(n // 2)
            yield torch.cat([mnist_x, aug_x], 0), torch.cat([mnist_y, aug_y], 0)
            current_n += n//2

class mnist_adv_aug(mnist_aug):
    def __init__(self, augname):
        super().__init__()
        with open(augname, "rb") as f:
            self.aug_data = torch.load(f)

    def adv_sample(self, n):
        idx = torch.randint(0, self.aug_data.shape[0], [n])
        return self.aug_data[idx], torch.zeros(n)

    def aug_sample(self, n):
        return torch.rand([n]+self.x_shape), torch.zeros(n)

    def train_sample(self, n):
        idx = torch.randint(0, self.mnist_train.data.shape[0], [n//2])
        mnist_x, mnist_y = self.data_preproc(self.mnist_train.data[idx]), self.target_preproc(self.mnist_train.targets[idx])
        adv_x, adv_y = self.adv_sample(n//4)
        aug_x, aug_y = self.aug_sample(n//4)
        return torch.cat([mnist_x, adv_x, aug_x], 0), torch.cat([mnist_y, adv_y, aug_y], 0)

    def test_sample(self, n):
        current_n = 0
        while current_n + n//2 < self.mnist_test.data.shape[0]:
            idx = torch.arange(current_n, current_n+n//2)
            mnist_x, mnist_y = self.data_preproc(self.mnist_test.data[idx]), self.target_preproc(self.mnist_test.targets[idx].float())
            aug_x, aug_y = self.aug_sample(n // 2)
            yield torch.cat([mnist_x, aug_x], 0), torch.cat([mnist_y, aug_y], 0)
            current_n += n//2

class count:
    def __init__(self, dataset_size=100000):
        self.data = torch.rand([dataset_size] + self.x_shape)
        self.targets = self.score_fn(self.data)
    @property
    def x_shape(self):
        return [1, 16, 16]
    @property
    def x_min(self):
        return 0.
    @property
    def x_max(self):
        return 1.
    @property
    def target_min(self):
        return 0.
    @property
    def target_max(self):
        return 100.
    def train_sample(self, n):
        idx = torch.randint(0, self.data.shape[0], [n])
        return self.data[idx], self.targets[idx]
    def test_sample(self, n):
        x = torch.rand([n] + self.x_shape)
        y = self.score_fn(x)
        yield x, y
    def score_fn(self, x):
        return 100 * ((1./(0.1 + (x - .5)**2) - 2.85) / (10 - 2.85)).sum(dim=[1,2,3]) / (16*16)

class masked_count(count):
    def __init__(self, dataset_size=100000):
        self.data = self.restriction_fn(torch.rand([dataset_size] + self.x_shape))
        self.targets = self.score_fn(self.data)
    @property
    def target_max(self):
        return 25.
    def score_fn(self, x):
        return super().score_fn(self.restriction_fn(x))
    def test_sample(self, n):
        x = self.restriction_fn(torch.rand([n] + self.x_shape))
        y = self.score_fn(x)
        yield x, y
    def restriction_fn(self, x):
        mask = torch.zeros_like(x)
        mask[...,4:12,4:12] = 1.
        return mask * x