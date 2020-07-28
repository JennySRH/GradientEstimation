import math

import torch
from torch import nn

from utils import GenerativeNet, InferenceNet, InputDependentBaseline, AutoRegressiveGenerativeNet, \
    AutoRegressiveInferenceNet, MaskedLinear, Bias, NonlinearGenerativeNet, NonlinearInferenceNet, gumbel_sigmoid, \
    log_bernoulli_prob, st_sigmoid


class SigmoidBeliefNetwork(nn.Module):
    def __init__(self,
                 mean_obs,
                 dim_hids=[200],
                 dim_obs=784,
                 use_nonlinear=False,
                 use_ar_gen=False,
                 use_ar_inf=True,
                 use_uniform_prior=False,
                 method='nvil',
                 temp=None,
                 num_samples=None):
        super(SigmoidBeliefNetwork, self).__init__()
        self.temp = temp
        self.num_layers = len(dim_hids)
        self.num_samples = num_samples
        gen_layers = dim_hids + [dim_obs]
        inf_layers = [dim_obs] + dim_hids[::-1]
        # z_L -> z_{L-1} -> ... -> z_1 -> x
        # x -> z_1 -> z_2 -> ... -> z_L
        if use_nonlinear:
            self.top_prior = Bias(gen_layers[0], trainable=use_uniform_prior)
            self.generative_net = NonlinearGenerativeNet(dim_obs, dim_hids[0])
            self.inference_net = NonlinearInferenceNet(dim_obs, dim_hids[0])
        else:
            if use_ar_gen and use_ar_inf:
                self.top_prior = MaskedLinear(torch.triu(torch.ones(gen_layers[0], gen_layers[0]), diagonal=1))
                self.generative_net = AutoRegressiveGenerativeNet(
                    *[nn.ModuleList([nn.Linear(gen_layers[i], gen_layers[i + 1]),
                                     MaskedLinear(
                                         torch.triu(torch.ones(gen_layers[i + 1], gen_layers[i + 1]), diagonal=1),
                                         bias=False)]) for i in range(len(gen_layers) - 1)])
                self.inference_net = AutoRegressiveInferenceNet(
                    *[nn.ModuleList([nn.Linear(inf_layers[i], inf_layers[i + 1]),
                                     MaskedLinear(
                                         torch.triu(torch.ones(inf_layers[i + 1], inf_layers[i + 1]), diagonal=1),
                                         bias=False)]) for i in range(len(inf_layers) - 1)])
            elif not use_ar_gen and use_ar_inf:
                self.top_prior = Bias(gen_layers[0])
                self.generative_net = GenerativeNet(
                    *[nn.Linear(gen_layers[i], gen_layers[i + 1]) for i in range(len(gen_layers) - 1)])
                self.inference_net = AutoRegressiveInferenceNet(
                    *[nn.ModuleList([nn.Linear(inf_layers[i], inf_layers[i + 1]),
                                     MaskedLinear(
                                         torch.triu(torch.ones(inf_layers[i + 1], inf_layers[i + 1]), diagonal=1),
                                         bias=False)]) for i in range(len(inf_layers) - 1)])
            elif use_ar_gen and not use_ar_inf:
                self.top_prior = MaskedLinear(torch.triu(torch.ones(gen_layers[0], gen_layers[0]), diagonal=1))
                self.generative_net = AutoRegressiveGenerativeNet(
                    *[nn.ModuleList([nn.Linear(gen_layers[i], gen_layers[i + 1]),
                                     MaskedLinear(
                                         torch.triu(torch.ones(gen_layers[i + 1], gen_layers[i + 1]), diagonal=1),
                                         bias=False)]) for i in range(len(gen_layers) - 1)])
                self.inference_net = InferenceNet(
                    *[nn.Linear(inf_layers[i], inf_layers[i + 1]) for i in range(len(inf_layers) - 1)])
            elif not use_ar_gen and not use_ar_inf:
                self.top_prior = Bias(gen_layers[0])
                self.generative_net = GenerativeNet(
                    *[nn.Linear(gen_layers[i], gen_layers[i + 1]) for i in range(len(gen_layers) - 1)])
                self.inference_net = InferenceNet(
                    *[nn.Linear(inf_layers[i], inf_layers[i + 1]) for i in range(len(inf_layers) - 1)])

        self.register_buffer('mean_obs', mean_obs)
        self.eval_step = self.compute_elbo
        if method == 'nvil':
            idb_dims = [dim_obs] + dim_hids[::-1][:-1]
            self.alpha = 0.8
            self.idb = nn.ModuleList([InputDependentBaseline(idb_dim, 100) for idb_dim in idb_dims])
            self.register_buffer('tri_1', torch.triu(torch.ones(self.num_layers, self.num_layers + 1)))
            self.register_buffer('tri_2', torch.triu(torch.ones(self.num_layers, self.num_layers)))
            self.register_buffer('running_mean', torch.zeros(self.num_layers))
            self.register_buffer('running_var', torch.zeros(self.num_layers))
            self.register_buffer('all_ones', torch.ones(self.num_layers))
            self.train_step = self.train_nvil
        else:
            # self.register_parameter('idb', None)
            # self.register_parameter('all_ones', None)
            # self.register_parameter('tri_2', None)
            # self.register_parameter('tri_1', None)
            # self.register_parameter('running_mean', None)
            # self.register_parameter('running_var', None)
            if method == 'vimco':
                self.train_step = self.train_vimco
                self.eval_step = self.compute_multisample_bound
            elif method == 'legrad':
                print('LeGrad algorithm only supports SBN with single hidden layer')
                self.train_step = self.train_legrad
            elif method == 'arm':
                self.train_step = self.train_arm
            elif method == 'rebar':
                self.train_step = self.train_rebar
            elif method == 'reinforce-loo':
                self.train_step = self.train_reinforce_loo
            elif method == 'st':
                self.train_step = self.train_st
                self.eval_step = self.eval_st
            elif method == 'gumbel':
                self.train_step = self.train_gumbel
                self.eval_step = self.eval_gumbel

    def compute_elbo(self, x, z=None):
        log_p_x_z, log_q_z, _ = self.compute_forward(x, z)
        sum_log_p_x_z = torch.sum(torch.stack(log_p_x_z, dim=1), dim=1)
        sum_log_q_z = torch.sum(torch.stack(log_q_z, dim=1), dim=1)
        # elbo shape : [batch_size]
        elbo = sum_log_p_x_z - sum_log_q_z
        return elbo
        
    def compute_forward(self, x, z=None, logits=None):
        log_p_x_z = []  # [p(z_1 | z_2), p(z_2 | z_3), ..., p(z_L)]
        log_q_z = []  # [q(z_1 | x), q(z_2 | z_1), q(z_3 | z_2) ..., q(z_L | z_{L-1})]

        # a list of tensors of multiple layers [z_1, z_2, ..., z_L],
        # each with shape [batch_size, latent_dim_i]
        # or [batch_size, num_samples, latent_dim_i]
        # sample z if not given
        # [z_1, z_2, ..., z_L]
        # the shape of z is [batch_size, latent_dim_i]
        # or [batch_size, num_samples, latent_dim_i]
        if z is None or logits is None:
            logits_q_z, samples_z = self.inference_net((x - self.mean_obs + 1.) / 2)
        if z is not None:
            samples_z = z
        if logits is not None:
            logits_q_z = logits
        # logit of [z_L , z_{L-1} , ...,  z_1, x] for prior
        logits_p_x_z = [self.top_prior(samples_z[-1])] + self.generative_net(samples_z[::-1] + [x])  

        log_p_x_z.append(log_bernoulli_prob(logits_p_x_z[-1], x))  # log likelihood 
        for i in range(self.num_layers):
            log_p_x_z.append(log_bernoulli_prob(logits_p_x_z[-(i + 2)], samples_z[i]))
            log_q_z.append(log_bernoulli_prob(logits_q_z[i], samples_z[i]))

        return log_p_x_z, log_q_z, (samples_z, logits_p_x_z, logits_q_z)

    def compute_multisample_bound(self, x, num_samples=None):
        num_samples = self.num_samples if num_samples is None else num_samples
        xx = x.unsqueeze(dim=1).repeat(1, num_samples, 1)
        log_p_x_z, log_q_z, _ = self.compute_forward(xx)
        log_p_x_z = torch.sum(torch.stack(log_p_x_z, dim=2), dim=2)
        log_q_z = torch.sum(torch.stack(log_q_z, dim=2), dim=2)
        estimated_nll = torch.logsumexp(log_p_x_z - log_q_z, dim=1) - math.log(num_samples)
        return estimated_nll

    def _vimco_compute_baseline(self, ls):
        s = torch.sum(ls, dim=1, keepdim=True)
        repeated_ls = ls.unsqueeze(dim=1).repeat(1, self.num_samples, 1)
        prev_baseline = repeated_ls + (1. / (self.num_samples - 1)) * torch.diag_embed(s - self.num_samples * ls)
        baseline = torch.logsumexp(prev_baseline, dim=2)  # [bs, num_samples]
        return baseline

    def train_vimco(self, x):
        xx = x.unsqueeze(dim=1).repeat(1, self.num_samples, 1)
        log_p_x_z, log_q_z, _ = self.compute_forward(xx)
        log_p_x_z = torch.sum(torch.stack(log_p_x_z, dim=2), dim=2)
        log_q_z = torch.sum(torch.stack(log_q_z, dim=2), dim=2)
        ls = log_p_x_z - log_q_z  # [bs, num_samples]
        elbo = torch.logsumexp(ls, dim=1, keepdim=True)
        with torch.no_grad():
            baselines = self._vimco_compute_baseline(ls)
            ws = (ls - elbo).exp()  # [bs, num_samples]
            learning_signal = elbo - baselines  # [bs, num_samples]
        loss_for_theta = (ws * log_p_x_z).sum(dim=1).mean().neg()
        loss_for_phi = (learning_signal * log_q_z - ws * log_q_z).sum(dim=1).mean().neg()
        total_loss = loss_for_theta + loss_for_phi
        return total_loss, (elbo - math.log(self.num_samples)).mean().neg().item()

    def _nvil_compute_idbs(self, layer_wise_input):
        baseline = []
        for i, fnn in enumerate(self.idb):
            baseline.append(fnn(layer_wise_input[i]).squeeze())
        return torch.stack(baseline, dim=1)

    def train_nvil(self, x):
        log_p_x_z, log_q_z, (samples_z, _, __) = self.compute_forward(x)
        log_p_x_z, log_q_z = torch.stack(log_p_x_z, dim=1), torch.stack(log_q_z, dim=1)
        idb_inputs = [(x - self.mean_obs + 1.) / 2] + samples_z
        idbs = self._nvil_compute_idbs(idb_inputs)
        with torch.no_grad():
            ls_per_layer = (torch.matmul(self.tri_1, log_p_x_z.unsqueeze(2)) -
                torch.matmul(self.tri_2, log_q_z.unsqueeze(2))).squeeze(dim=2)
            ls_sub_idb = ls_per_layer - idbs
            elbo = log_p_x_z.sum(dim=1) - log_q_z.sum(dim=1)
            # assign less weight on the first batch
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * torch.mean(ls_sub_idb, dim=0,
                                                                                               keepdim=True)
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * torch.var(ls_sub_idb, dim=0,
                                                                                            keepdim=True)
            learning_signal = (ls_sub_idb - self.running_mean) / torch.max(self.all_ones, self.running_var.sqrt())
        
        loss_for_theta = log_p_x_z.sum(dim=1).mean().neg()
        loss_for_phi = (learning_signal * log_q_z).sum(dim=1).mean().neg()
        loss_for_psi = (learning_signal * idbs).sum(dim=1).mean().neg()
        total_loss = loss_for_theta + loss_for_phi + loss_for_psi
        return total_loss, elbo.mean().neg().item()

    def _compute_flipped_f(self, x, logit_q, sampled_h):
        p_xh_0 = []
        p_xh_1 = []
        q_h_0 = []
        q_h_1 = []
        for i in range(self.num_layers):
            dim = sampled_h[i].shape[-1]
            cur_h = sampled_h[i].unsqueeze(dim=1).repeat(1, dim, 1)
            cur_h.diagonal(dim1=-2, dim2=-1).copy_(torch.zeros(dim))
            logit_px_h = self.generative_net([cur_h, x])
            logit_p = self.top_prior(cur_h)
            p_xh_0 = log_bernoulli_prob(logit_p.unsqueeze(dim=1), cur_h) + \
                     log_bernoulli_prob(logit_px_h[-1], x)
            q_h_0 = log_bernoulli_prob(logit_q[i].unsqueeze(dim=1), cur_h)

            cur_h.diagonal(dim1=-2, dim2=-1).copy_(torch.ones(dim))
            logit_px_h = self.generative_net([cur_h, x])
            logit_p = self.top_prior(cur_h)
            p_xh_1 = log_bernoulli_prob(logit_p.unsqueeze(dim=1), cur_h) + \
                     log_bernoulli_prob(logit_px_h[-1], x)
            q_h_1 = log_bernoulli_prob(logit_q[i].unsqueeze(dim=1), cur_h)
        return p_xh_0, q_h_0, p_xh_1, q_h_1

    def train_legrad(self, x):
        logit_q, sampled_h = self.inference_net((x - self.mean_obs + 1.) / 2)
        # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_p_x_h = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_p = [self.top_prior(sampled_h[-1])] + logit_p_x_h[:-1]
        log_ll = log_bernoulli_prob(logit_p_x_h[-1], x).unsqueeze(dim=1)  # log likelihood
        log_qh_x = log_bernoulli_prob(logit_q[0], sampled_h[0]).unsqueeze(dim=1)
        log_ph = log_bernoulli_prob(logit_p[-1], sampled_h[0]).unsqueeze(dim=1)
        with torch.no_grad():
            p_xh_0, q_h_0, p_xh_1, q_h_1 = self._compute_flipped_f(x.unsqueeze(dim=1), logit_q, sampled_h)
            elbo = (log_ph + log_ll - log_qh_x).squeeze().mean().neg().item()
        total_loss = (log_ph + log_ll).squeeze().mean().neg()

        for i in range(self.num_layers):
            total_loss += torch.sum(torch.sigmoid(logit_q[i]) * (p_xh_1 - q_h_1) +
                                    torch.sigmoid(-logit_q[i]) * (p_xh_0 - q_h_0),
                                    dim=-1).mean().neg()
        return total_loss, elbo

    def _forward_from_middle_layers(self, x, z, layer, samples_z, logits_q_z):
        # pass the binary vector at layer t 
        # and return all logits and samples at i-th layer, for all i > t.
        logits_q_gt_t, samples_z_gt_t = self.inference_net.manual_forward(z, layer + 1)

        # re-combine all logits and samples 1-{t-1}, t, {t+1}-T
        samples_z = samples_z[0:layer] + [z] + samples_z_gt_t
        logits_q = logits_q_z[0:layer + 1] + logits_q_gt_t

        # compute elbo
        log_p_x_z, log_q_z, _ = self.compute_forward(x, samples_z, logits_q)
        elbo = torch.sum(torch.stack(log_p_x_z, dim=1), dim=1) - torch.sum(torch.stack(log_q_z, dim=1), dim=1)
        return elbo
        
    def train_arm(self, x):
        # list with order [z_1, z_2, ..., z_L]
        log_p_x_z, log_q_z, (samples_z, logits_p_x_z, logits_q_z) = self.compute_forward(x)
        log_p_x_z = torch.sum(torch.stack(log_p_x_z, dim=1),dim=1)
        log_q_z = torch.sum(torch.stack(log_q_z, dim=1),dim=1)
        loss_for_theta = 0.0
        with torch.no_grad():
            elbo = (log_p_x_z - log_q_z).mean().neg().item()
        for layer in range(self.num_layers):
            with torch.no_grad():
                u = torch.rand_like(logits_q_z[layer])
                prob = torch.sigmoid(logits_q_z[layer])
                z_1 = (u > (1. - prob)).float()
                z_2 = (u < prob).float()
                f_1 = self._forward_from_middle_layers(x, z_1, layer, samples_z, logits_q_z).unsqueeze(1)
                f_2 = self._forward_from_middle_layers(x, z_2, layer, samples_z, logits_q_z).unsqueeze(1)
            loss_for_theta += torch.sum((f_1 - f_2) * (u - 0.5) * logits_q_z[layer], dim=-1)
        
        total_loss = (log_p_x_z + loss_for_theta).mean().neg()
        return total_loss, elbo

    def train_disarm(self, x):
        # list with order [z_1, z_2, ..., z_L]
        log_p_x_z, log_q_z, (samples_z, logits_p_x_z, logits_q_z) = self.compute_forward(x)
        log_p_x_z = torch.sum(torch.stack(log_p_x_z, dim=1),dim=1)
        log_q_z = torch.sum(torch.stack(log_q_z, dim=1),dim=1)
        loss_for_theta = 0.0
        with torch.no_grad():
            elbo = (log_p_x_z - log_q_z).mean().neg().item()
        for layer in range(self.num_layers):
            with torch.no_grad():
                u = torch.rand_like(logits_q_z[layer])
                prob = torch.sigmoid(logits_q_z[layer])
                abs_prob = torch.sigmoid(torch.abs(logits_q_z[layer]))
                z_1 = (u > (1. - prob)).float()
                z_2 = (u < prob).float()
                f_1 = self._forward_from_middle_layers(x, z_1, layer, samples_z, logits_q_z).unsqueeze(1)
                f_2 = self._forward_from_middle_layers(x, z_2, layer, samples_z, logits_q_z).unsqueeze(1)
                learning_signal = 0.5 * (f1 - f2) * (torch.square(z_1 - z_2)) * (-1.)**z_2 * abs_prob
            loss_for_theta += torch.sum(learning_signal * logits_q_z[layer], dim=-1)
        
        total_loss = (log_p_x_z + loss_for_theta).mean().neg()
        return total_loss, elbo

    def train_reinforce_loo(self, x):
        # list with order [z_1, z_2, ..., z_L]
        log_p_x_z, log_q_z, (samples_z, logits_p_x_z, logits_q_z) = self.compute_forward(x)
        log_p_x_z = torch.sum(torch.stack(log_p_x_z, dim=1),dim=1)
        log_q_z = torch.sum(torch.stack(log_q_z, dim=1),dim=1)
        loss_for_theta = 0.0
        with torch.no_grad():
            elbo = (log_p_x_z - log_q_z).mean().neg().item()
        for layer in range(self.num_layers):
            with torch.no_grad():
                u_1 = torch.rand_like(logits_q_z[layer])
                u_2 = torch.rand_like(logits_q_z[layer])
                prob = torch.sigmoid(logits_q_z[layer])
                z_1 = (u_1 < prob).float()
                z_2 = (u_2 < prob).float()
                f_1 = self._forward_from_middle_layers(x, z_1, layer, samples_z, logits_q_z)
                f_2 = self._forward_from_middle_layers(x, z_2, layer, samples_z, logits_q_z)
            log_q_z_1 = log_bernoulli_prob(logits_q_z[layer], z_1)
            log_q_z_2 = log_bernoulli_prob(logits_q_z[layer], z_2)
            loss_for_theta += 0.5 * (f_1 - f_2) * log_q_z_1 + 0.5 * (f_2 - f_1) * log_q_z_2
        
        total_loss = (log_p_x_z + loss_for_theta).mean().neg()
        return total_loss, elbo
    def _reparam_z_z_tilde(self, logit, eps=1e-6):
        # adapted from https://github.com/duvenaud/relax/blob/master/rebar_tf.py
        # noise for generating z
        # logit : [bs. dim]
        u = torch.rand_like(logit).clamp(min=eps, max=1. - eps)
        # logistic reparameterization z = g(u, log_alpha)
        z = logit + torch.log(u) - torch.log(1.0 - u)
        # b = H(z)
        b = (z > 0.0).float()
        # g(u', log_alpha) = 0
        u_prime = torch.sigmoid(-logit).clamp(min=eps, max=1. - eps)
        with torch.no_grad():
            v_1 = (u - u_prime) / (1. - u_prime)
            v_0 = u / u_prime
        v_1 = v_1.detach() * (1 - u_prime) + u_prime
        v_0 = v_0.detach() * u_prime
        v = (v_1 * b + v_0 * (1. - b)).clamp(min=eps, max=1. - eps)
        z_tilde = logit + torch.log(v) - torch.log(1.0 - v)
        return b, z, z_tilde

    def eval_gumbel(self, x):
        logit_q, _ = self.inference_net((x - self.mean_obs + 1.) / 2)
        sampled_h = [gumbel_sigmoid(logit_q[0], self.temp)]
        # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_pxh = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_p = [self.top_prior(sampled_h[-1])] + logit_pxh[:-1]
        log_px_h = log_bernoulli_prob(logit_pxh[-1], x)
        log_ph = log_bernoulli_prob(logit_p[-1], sampled_h[0])
        log_qh_x = log_bernoulli_prob(logit_q[0], sampled_h[0])
        elbo = (log_ph + log_px_h - log_qh_x).mean().neg()
        return elbo.item()

    def train_gumbel(self, x):
        logit_q, _ = self.inference_net((x - self.mean_obs + 1.) / 2)
        sampled_h = [gumbel_sigmoid(logit_q[0], self.temp)]
        # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_pxh = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_p = [self.top_prior(sampled_h[-1])] + logit_pxh[:-1]
        log_px_h = log_bernoulli_prob(logit_pxh[-1], x)
        log_ph = log_bernoulli_prob(logit_p[-1], sampled_h[0])
        log_qh_x = log_bernoulli_prob(logit_q[0], sampled_h[0])
        elbo = (log_ph + log_px_h - log_qh_x).mean().neg()
        return elbo, elbo.item()

    def train_st(self, x):
        logit_q, _ = self.inference_net((x - self.mean_obs + 1.) / 2)
        sampled_h = [st_sigmoid(logit_q[0])]
        # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_pxh = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_p = [self.top_prior(sampled_h[-1])] + logit_pxh[:-1]
        log_px_h = log_bernoulli_prob(logit_pxh[-1], x)
        log_ph = log_bernoulli_prob(logit_p[-1], sampled_h[0])
        log_qh_x = log_bernoulli_prob(logit_q[0], sampled_h[0])
        elbo = (log_ph + log_px_h - log_qh_x).mean().neg()
        return elbo, elbo.item()

    def eval_st(self, x):
        logit_q, _ = self.inference_net((x - self.mean_obs + 1.) / 2)
        sampled_h = [st_sigmoid(logit_q[0], self.temp)]
        # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_pxh = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_p = [self.top_prior(sampled_h[-1])] + logit_pxh[:-1]
        log_px_h = log_bernoulli_prob(logit_pxh[-1], x)
        log_ph = log_bernoulli_prob(logit_p[-1], sampled_h[0])
        log_qh_x = log_bernoulli_prob(logit_q[0], sampled_h[0])
        elbo = (log_ph + log_px_h - log_qh_x).mean().neg()
        return elbo.item()

    def train_rebar(self, x):
        logit_q, _ = self.inference_net((x - self.mean_obs + 1.) / 2)
        b, z, z_tilde = self._reparam_z_z_tilde(logit_q[0])
        logit_pxh = self.generative_net([b] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_p = [self.top_prior(b)] + logit_pxh[:-1]
        log_px_h = log_bernoulli_prob(logit_pxh[-1], x)
        log_ph = log_bernoulli_prob(logit_p[-1], b)
        log_qh_x = log_bernoulli_prob(logit_q[0], b)
        f_z = self._compute_fxh(x, torch.sigmoid(z / self.temp), logit_q[0]).unsqueeze(dim=1)
        f_z_tilde = self._compute_fxh(x, torch.sigmoid(z_tilde / self.temp), logit_q[0]).unsqueeze(dim=1)
        with torch.no_grad():
            f_b = (log_ph + log_px_h - log_qh_x).unsqueeze(dim=1)
        # print("f_z :", torch.mean(f_z))
        # print("f_z_tilde :", torch.mean(f_z_tilde))
        # print("f_b :", torch.mean(f_b))
        # print("log_qh_x :", torch.mean(log_qh_x))
        loss_for_theta = torch.sum((f_b - f_z_tilde.detach()) * log_qh_x.unsqueeze(dim=1) - f_z_tilde + f_z, dim=-1)
        total_loss = (log_ph + log_px_h + loss_for_theta).mean().neg()
        # print("debug #####################################################")
        # print(((f_b - f_z_tilde.detach()) * log_qh_x.unsqueeze(dim=1)).shape)
        # print(f_z_tilde.shape)
        # print(f_z.shape)
        # print(log_ph.shape)
        # print(log_px_h.shape)
        return total_loss, f_b.mean().neg().item()
