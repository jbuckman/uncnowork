import torch, math
from torch import nn

class EnsembleLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features', 'ens_size']

    def __init__(self, in_features, out_features, ens_size, bias=True):
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ens_size = ens_size
        self.weight = nn.Parameter(torch.Tensor(ens_size, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, ens_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, -1/math.sqrt(5), 1/math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -1/math.sqrt(5), 1/math.sqrt(5))

    def forward(self, input, expand=False):
        if expand:
            if len(input.shape) == 2:
                out = torch.einsum("bi,mio->bmo", input, self.weight)
                if self.bias is not None: out += self.bias
            elif len(input.shape) == 1:
                out = torch.einsum("i,mio->mo", input, self.weight)
                if self.bias is not None: out += self.bias[0]
            else: raise Exception("input shape error")
        else:
            if len(input.shape) == 3:
                out = torch.einsum("bmi,mio->bmo", input, self.weight)
                if self.bias is not None: out += self.bias
            elif len(input.shape) == 2:
                out = torch.einsum("mi,mio->mo", input, self.weight)
                if self.bias is not None: out += self.bias[0]
            else: raise Exception("input shape error")
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, ens_size={}, bias={}'.format(
            self.in_features, self.out_features, self.ens_size, self.bias is not None
        )

class RegularNet(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        if output_dim == 0: output_dim = 1; self.squeeze_out=True
        else: self.squeeze_out=False
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nonlin = nn.ReLU(inplace=True)
        self.layers = torch.nn.ModuleList([
            nn.Linear(in_d, out_d)
            for in_d, out_d in zip((input_dim,)+tuple(layers),tuple(layers)+(output_dim,))])

    def forward(self, x):
        x = self.nonlin(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.nonlin(layer(x))
        x = self.layers[-1](x)
        if self.squeeze_out: x = x[...,0]
        return x

def conv2d_out_shape(in_shape, conv2d):
    C = conv2d.out_channels
    H = math.floor((in_shape[-2] + 2.*conv2d.padding[0] - conv2d.dilation[0] * (conv2d.kernel_size[0] - 1.) - 1.)/conv2d.stride[0] + 1.)
    W = math.floor((in_shape[-1] + 2.*conv2d.padding[1] - conv2d.dilation[1] * (conv2d.kernel_size[1] - 1.) - 1.)/conv2d.stride[1] + 1.)
    return [C, H, W]
cs = conv2d_out_shape

class ConvNet(nn.Module):
    def __init__(self, input_shape, output_dim, layers):
        if output_dim == 0: output_dim = 1; self.squeeze_out=True
        else: self.squeeze_out=False
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=layers[0],
            kernel_size=8,
            stride=3,
            padding=4
        )
        self.conv2 = nn.Conv2d(layers[0], layers[1], kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(layers[1], layers[2], kernel_size=2, stride=1)
        self.shape_after_convs = cs(cs(cs(input_shape, self.conv1), self.conv2), self.conv3)
        self.fc = RegularNet(self.shape_after_convs[0]*self.shape_after_convs[1]*self.shape_after_convs[2], output_dim, layers=layers[3:])

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = x.view((x.shape[0], -1,))
        out = self.fc(x)
        if self.squeeze_out: out = out[...,0]
        return out

class UncertainNet(nn.Module):
    def __init__(self, input_dim, output_dim, ensemble_n, layers):
        if output_dim == 0: output_dim = 1; self.squeeze_out=True
        else: self.squeeze_out=False
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_n = ensemble_n
        self.nonlin = nn.ReLU(inplace=True)
        self.layers = torch.nn.ModuleList([
            EnsembleLinear(in_d, out_d, 1 + ensemble_n)
            for in_d, out_d in zip((input_dim,)+tuple(layers),tuple(layers)+(output_dim,))])

    def forward(self, x, separate_ensemble=False, upperbound=False, α=None, β=None):
        x = self.nonlin(self.layers[0](x, expand=True))
        for layer in self.layers[1:-1]:
            x = self.nonlin(layer(x))
        x = self.layers[-1](x)
        main_x = x[:,0]
        ens_x = x[:,1:]
        if separate_ensemble:
            if self.squeeze_out: main_x = main_x[...,0]; ens_x = ens_x[...,0]
            return main_x, ens_x
        else:
            L = (ens_x - main_x[:,None,:])
            if upperbound:
                t_α2_x = L.kthvalue(1 + int(α * self.ensemble_n), axis=1)[0]
                ans = main_x - β * t_α2_x
            else:
                t_1α2_x = L.kthvalue(int((1. - α) * self.ensemble_n), axis=1)[0]
                ans = main_x - β * t_1α2_x
            if self.squeeze_out: ans = ans[...,0]
            return ans

class UncertainConvNet(ConvNet):
    def __init__(self, input_shape, output_dim, ensemble_n, layers):
        super().__init__(input_shape, output_dim, layers)
        self.fc = UncertainNet(self.shape_after_convs[0]*self.shape_after_convs[1]*self.shape_after_convs[2], output_dim, ensemble_n, layers=layers[3:])

    def forward(self, x, separate_ensemble=False, upperbound=False, α=None, β=None):
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = x.view((x.shape[0], -1,))
        x = self.fc(x, separate_ensemble, upperbound, α, β)
        return x
