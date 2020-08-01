import math

import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torch.nn import Module
from torch.nn import init
from torch.nn.functional import softplus, softmax


class InputDependentBaseline(Module):
    def __init__(self, input_dim, aux_net_dim=100):
        super(InputDependentBaseline, self).__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, aux_net_dim),
            nn.Tanh(),
            nn.Linear(aux_net_dim, 1)
        )

    def forward(self, x):
        return self.fnn(2 * x - 1.)


class NonlinearInferenceNet(Module):
    def __init__(self, dim_observed, dim_latent):
        super(NonlinearInferenceNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.3)
        self.enc_fc1 = nn.Linear(dim_observed, dim_latent)
        self.enc_fc2 = nn.Linear(dim_latent, dim_latent)
        self.enc_fc3 = nn.Linear(dim_latent, dim_latent)

    def forward(self, x):
        logits = []
        h = []
        h1 = self.lrelu(self.enc_fc1(2 * x - 1.))
        h2 = self.lrelu(self.enc_fc2(h1))
        logit = self.enc_fc3(h2)
        u = torch.rand_like(logit)
        x = (torch.sigmoid(logit) > u).float()
        logits.append(logit)
        h.append(x)
        return logits, h

    def manually_forward(self, x, starting_layer=None, reuse_u=False, global_u=None):
        logits = []
        samples_z = []
        if reuse_u:
            U = global_u
        else:
            U = []
        if starting_layer is not None:
            return logits, samples_z, U

        h1 = self.lrelu(self.enc_fc1(2 * x - 1.))
        h2 = self.lrelu(self.enc_fc2(h1))
        logit = self.enc_fc3(h2)
        if reuse_u:
            u = U[idx]
        else:
            u = torch.rand_like(logit)
            U.append(u) 
        x = (torch.sigmoid(logit) > u).float()
        logits.append(logit)
        samples_z.append(x)
        return logits, samples_z, U

class AutoRegressiveInferenceNet(Module):
    def __init__(self, *args):
        super(AutoRegressiveInferenceNet, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x):
        logits = []
        h = []
        for module in self._modules.values():
            logit = module[0](2 * x - 1.)
            with torch.no_grad():
                ar_weight = buf = torch.zeros_like(logit.detach())
                for i in range(logit.shape[-1]):
                    ar_weight[..., i] = torch.sum(module[1].weight[i] * buf, dim=-1)
                    single_unit = logit[..., i] + ar_weight[..., i]
                    u = torch.rand_like(single_unit)
                    buf[..., i] = (torch.sigmoid(single_unit) > u).float()
            logits.append(logit + module[1](2 * buf - 1.))
            h.append(buf)
        return logits, h


class InferenceNet(Module):
    def __init__(self, *args):
        super(InferenceNet, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, x):
        logits = []
        samples_z = []
        for idx, module in enumerate(self._modules.values()):
            logit = module(2 * x - 1.)
            u = torch.rand_like(logit)
            x = (torch.sigmoid(logit) > u).float()
            logits.append(logit)
            samples_z.append(x)
        return logits, samples_z

    def manually_forward(self, x, starting_layer=None, reuse_u=False, global_u=None):
        logits = []
        samples_z = []
        if reuse_u:
            U = global_u
        else:
            U = []
        for idx, module in enumerate(self._modules.values()):
            if starting_layer is not None and idx < starting_layer:
                continue
            logit = module(2 * x - 1.)
            if reuse_u:
                u = U[idx]
            else:
                u = torch.rand_like(logit)
                U.append(u) 
            # print(logit.shape)
            # print(u.shape)
            x = (torch.sigmoid(logit) > u).float()
            logits.append(logit)
            samples_z.append(x)
            
        return logits, samples_z, U



class NonlinearGenerativeNet(Module):
    def __init__(self, dim_observed, dim_latent, bias_init=None):
        super(NonlinearGenerativeNet, self).__init__()
        self.lrelu = nn.LeakyReLU(0.3)
        self.dec_fc1 = nn.Linear(dim_latent, dim_latent)
        self.dec_fc2 = nn.Linear(dim_latent, dim_latent)
        self.dec_fc3 = nn.Linear(dim_latent, dim_observed)
        if bias_init is not None:
            with torch.no_grad():
                self.dec_fc3.bias = nn.Parameter(bias_init)

    def forward(self, x):
        logits = []
        h1 = self.lrelu(self.dec_fc1(2 * x[0] - 1.))
        h2 = self.lrelu(self.dec_fc2(h1))
        logit = self.dec_fc3(h2)
        logits.append(logit)
        return logits

class AutoRegressiveGenerativeNet(Module):
    def __init__(self, *args, bias_init=None):
        super(AutoRegressiveGenerativeNet, self).__init__()
        for idx, module in enumerate(args):
            if idx == len(args) - 1 and bias_init is not None:
                with torch.no_grad():
                    module.bias = nn.Parameter(bias_init)
            self.add_module(str(idx), module)
            
    def forward(self, x):
        list = []
        for idx, module in enumerate(self._modules.values()):
            logit = module[0](2 * x[idx] - 1.) + module[1](2 * x[idx + 1] - 1.)
            list.append(logit)
        return list


class GenerativeNet(Module):
    def __init__(self, *args, bias_init=None):
        super(GenerativeNet, self).__init__()
        for idx, module in enumerate(args):
            if idx == len(args) - 1 and bias_init is not None:
                with torch.no_grad():
                    module.bias = nn.Parameter(bias_init)
            self.add_module(str(idx), module)

    def forward(self, x):
        logits = []
        for idx, module in enumerate(self._modules.values()):
            logit = module(2 * x[idx] - 1.)
            logits.append(logit)
        return logits


class MultiOptim(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()

    def step(self):
        for optim in self.optimizers:
            optim.step()

    def adjust_lr(self):
        for optimizer in self.optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.7


def log_bernoulli_prob(logits, input):
    """
    output shape : [batch_size]
    """
    return torch.sum(logits * input, dim=-1) - torch.sum(softplus(logits), dim=-1)


def gumbel_sigmoid(logits, temperature, hard=True, eps=1e-20):
    u = torch.rand_like(logits)
    logistic_noise = torch.log(u + eps) - torch.log(1 - u + eps)
    y = logits + logistic_noise
    sampled = torch.sigmoid(y / temperature)
    # sampled = torch.clamp((y + 1.) / (2. * temperature), 0., 1.)
    if not hard:
        return sampled
    hard_samples = (sampled > 0.5).float()
    return (hard_samples - sampled).detach() + sampled


def gumbel_softmax(logits, temperature, hard=False, eps=1e-20):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    sampled = softmax(y / temperature, dim=-1)

    if not hard:
        return sampled.view(-1, logits.shape[-1] * logits.shape[-2])

    shape = sampled.size()
    _, ind = sampled.max(dim=-1)
    y_hard = torch.zeros_like(sampled).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - sampled).detach() + sampled
    return y_hard.view(-1, logits.shape[-1] * logits.shape[-2])


def st_sigmoid(logit, is_logit=True):
    shape = logit.size()
    if is_logit:
        prob = torch.sigmoid(logit)
        # prob = torch.clamp((logit + 1.)/2., 0., 1.)
    else:
        prob = logit
    u = torch.rand_like(prob)
    output_binary = prob.data.new(*shape).zero_().add(((prob - u) > 0.).float())
    output = (output_binary - prob).detach() + prob
    return output


# adapted from
# https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
class MaskedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        if input.dim() == 2 and bias is not None:
            # fused op is marginally faster
            ret = torch.addmm(bias, input, weight.t())
        else:
            output = input.matmul(weight.t())
            if bias is not None:
                output += bias
            ret = output
        ctx.save_for_backward(input, weight, bias, mask)
        return ret

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(dim0=-2, dim1=-1).mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        # if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask


class MaskedLinear(Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Argumens
        ------------------
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(MaskedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return MaskedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )


class Bias(Module):
    def __init__(self, dim, trainable=True):
        super(Bias, self).__init__()
        if trainable:
            self.bias = nn.Parameter(torch.Tensor(1, dim))
            self.reset_parameters()
        else:
            self.register_buffer('bias', torch.zeros(1, dim))

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.bias.shape[1])
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return self.bias