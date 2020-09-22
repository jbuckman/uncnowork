import torch
from nn import *

class vanilla:
    def __init__(self, task, layers=(32, 32, 64, 64), **kwargs):
        if len(task.x_shape) == 1:
            self.nn = RegularNet(input_dim=task.x_shape[0], output_dim=0, layers=layers)
        else:
            self.nn = ConvNet(input_shape=task.x_shape, output_dim=0, layers=layers)
        self.opt = torch.optim.Adam(self.nn.parameters(), 1e-4)

    def learn(self, step, x, y):
        """
        :param step: int, how many steps of training
        :param idx: [n] of ints, a unique id per data point
        :param x: [n, x_shape...] of inputs
        :param a: [n] of ints corresponding to action selections
        :param y: [n] of floats, the targets for each (x,a)
        :return: loss at this step
        """
        guess = self.nn(x)
        loss = torch.nn.functional.mse_loss(guess, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(loss=loss.detach().item())

    def predict(self, x):
        """
        :param x: [n, x_shape...] of test inputs
        :return: [n, n_arms] of predictions for each arm
        """
        return self.nn(x)

class quantile:
    def __init__(self, task, layers=(32, 32, 64, 64), α=.1, **kwargs):
        if len(task.x_shape) == 1:
            self.nn = RegularNet(input_dim=task.x_shape[0], output_dim=0, layers=layers)
        else:
            self.nn = ConvNet(input_shape=task.x_shape, output_dim=0, layers=layers)
        self.opt = torch.optim.Adam(self.nn.parameters(), 1e-4)
        self.α = α

    def learn(self, step, x, y):
        """
        :param step: int, how many steps of training
        :param idx: [n] of ints, a unique id per data point
        :param x: [n, x_shape...] of inputs
        :param a: [n] of ints corresponding to action selections
        :param y: [n] of floats, the targets for each (x,a)
        :return: loss at this step
        """
        guess = self.nn(x)
        losses = torch.nn.functional.mse_loss(guess, y, reduction="none")
        loss = ((losses[guess < y] * self.α).sum() + (losses[guess > y] * (1. - self.α)).sum()) / x.shape[0]
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return dict(loss=loss.detach().item())

    def predict(self, x):
        """
        :param x: [n, x_shape...] of test inputs
        :return: [n, n_arms] of predictions for each arm
        """
        return self.nn(x)

class ensemble:
    def __init__(self, task, layers=(32, 32, 64, 64),
                 ensemble_n=10, α=.1, β=1., subsample_rate=.5, **kwargs):
        if len(task.x_shape) == 1:
            self.nn = UncertainNet(input_dim=task.x_shape[0], output_dim=0, layers=layers, ensemble_n=ensemble_n)
        else:
            self.nn = UncertainConvNet(input_shape=task.x_shape, output_dim=0, layers=layers, ensemble_n=ensemble_n)
        self.opt = torch.optim.Adam(self.nn.parameters(), 1e-4)
        self.ensemble_n = ensemble_n
        self.α = α
        self.β = β
        self.subsample_rate = subsample_rate

    def learn(self, step, x, y):
        """
        :param step: int, how many steps of training
        :param idx: [n] of ints, a unique id per data point
        :param x: [n, x_shape...] of inputs
        :param a: [n] of ints corresponding to action selections
        :param y: [n] of floats, the targets for each (x,a)
        :return: loss at this step
        """
        mask = torch.rand([x.shape[0], self.ensemble_n]) < self.subsample_rate
        main_guess, ens_guess = self.nn(x, separate_ensemble=True)
        main_loss = torch.nn.functional.mse_loss(main_guess, y)
        ens_loss = torch.nn.functional.mse_loss(ens_guess[mask], y[:,None].repeat(1,self.ensemble_n)[mask])
        self.opt.zero_grad()
        (main_loss + ens_loss).backward()
        self.opt.step()
        return dict(main_loss=main_loss.detach().item(),
                    ens_loss=ens_loss.detach().item())

    def predict(self, x):
        """
        :param x: [n, x_shape...] of test inputs
        :return: [n, n_arms] of predictions for each arm
        """
        return self.nn(x, α=self.α, β=self.β)

class rnd:
    def __init__(self, task, layers=(32, 32, 64, 64), α=1., **kwargs):
        self.α = α
        if len(task.x_shape) == 1:
            self.nn = RegularNet(input_dim=task.x_shape[0], output_dim=0, layers=layers)
            self.random_nn = RegularNet(input_dim=task.x_shape[0], output_dim=0, layers=(128, 128, 128, 128))
            self.copier_nn = RegularNet(input_dim=task.x_shape[0], output_dim=0, layers=(2, 2, 2, 8))
        else:
            self.nn = ConvNet(input_shape=task.x_shape, output_dim=0, layers=layers)
            self.random_nn = ConvNet(input_shape=task.x_shape, output_dim=0, layers=(128, 128, 128, 128))
            self.copier_nn = ConvNet(input_shape=task.x_shape, output_dim=0, layers=(2, 2, 2, 8))
        self.opt = torch.optim.Adam(list(self.nn.parameters()) + list(self.copier_nn.parameters()), 1e-4)

    def learn(self, step, x, y):
        """
        :param step: int, how many steps of training
        :param idx: [n] of ints, a unique id per data point
        :param x: [n, x_shape...] of inputs
        :param a: [n] of ints corresponding to action selections
        :param y: [n] of floats, the targets for each (x,a)
        :return: loss at this step
        """
        guess = self.nn(x)
        loss = torch.nn.functional.mse_loss(guess, y)
        random_y = (self.random_nn(x) * 100).detach()
        copy_guess = self.copier_nn(x)
        copy_loss = torch.nn.functional.mse_loss(copy_guess, random_y)
        self.opt.zero_grad()
        (loss + copy_loss).backward()
        self.opt.step()
        return dict(loss=loss.detach().item(), copy_loss=copy_loss.detach().item())

    def predict(self, x):
        """
        :param x: [n, x_shape...] of test inputs
        :return: [n, n_arms] of predictions for each arm
        """
        return self.nn(x) - self.α * (self.copier_nn(x) - self.random_nn(x)*100)**2