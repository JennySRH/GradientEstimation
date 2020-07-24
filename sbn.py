import math

import torch
from torch import nn

from utils import GenerativeNet, InferenceNet, InputDependentBaseline, AutoRegressiveGenerativeNet, \
    AutoRegressiveInferenceNet, MaskedLinear, Bias, NonlinearGenerativeNet, NonlinearInferenceNet, gumbel_sigmoid, \
    log_bernoulli_prob, st_sigmoid


class SBN(nn.Module):
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
        super(SBN, self).__init__()
        self.temp = temp
        self.num_layers = len(dim_hids)
        self.num_samples = num_samples
        gen_layers = dim_hids + [dim_obs]
        inf_layers = [dim_obs] + dim_hids[::-1]
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
        if method == 'nvil':
            idb_dims = [dim_obs] + dim_hids[::-1][:-1]
            self.alpha = 0.8
            self.idb = nn.ModuleList([InputDependentBaseline(idb_dim, 100) for idb_dim in idb_dims])
            self.register_buffer('tri_1', torch.triu(torch.ones(self.num_layers, self.num_layers + 1)))
            self.register_buffer('tri_2', torch.triu(torch.ones(self.num_layers, self.num_layers)))
            self.register_buffer('all_ones', torch.ones(self.num_layers))
            self.register_buffer('running_mean', torch.zeros(self.num_layers))
            self.register_buffer('running_var', torch.zeros(self.num_layers))
            self.train_step = self.train_nvil
            self.eval_step = self.eval_single_sample_elbo
        else:
            self.register_parameter('idb', None)
            self.register_parameter('all_ones', None)
            self.register_parameter('tri_2', None)
            self.register_parameter('tri_1', None)
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            if method == 'vimco':
                self.train_step = self.train_vimco
                self.eval_step = self.eval_multi_sample_elbo
            elif method == 'legrad':
                print('LeGrad algorithm only supports SBN with single hidden layer')
                self.train_step = self.train_legrad
                self.eval_step = self.eval_single_sample_elbo
            elif method == 'arm':
                self.train_step = self.train_arm
                self.eval_step = self.eval_single_sample_elbo
            elif method == 'rebar':
                self.train_step = self.train_rebar
                self.eval_step = self.eval_single_sample_elbo
            elif method == 'st':
                self.train_step = self.train_st
                self.eval_step = self.eval_st
            elif method == 'gumbel':
                self.train_step = self.train_gumbel
                self.eval_step = self.eval_gumbel

    def compute_logqh_x_and_logpxh(self, x):
        log_p = 0.
        log_q = 0.
        logit_q, sampled_h = self.inference_net(
            (x - self.mean_obs + 1.) / 2)  # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_p_x_h = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        log_ll = log_bernoulli_prob(logit_p_x_h[-1], x)  # log likelihood
        logit_p = [self.top_prior(sampled_h[-1])] + logit_p_x_h[:-1]  # logit of [h_n , h_n-1 , ...,  h_1] for prior
        for i in range(self.num_layers):
            log_p = log_p + log_bernoulli_prob(logit_p[-(i + 1)], sampled_h[i])
            log_q = log_q + log_bernoulli_prob(logit_q[i], sampled_h[i])
        return log_p + log_ll, log_q

    def compute_nll(self, x, test_samples=1000):
        xx = x.unsqueeze(dim=1).repeat(1, test_samples, 1)
        log_pxh, log_qh_x = self.compute_logqh_x_and_logpxh(xx)
        estimated_nll = torch.logsumexp(log_pxh - log_qh_x, dim=1) - math.log(test_samples)
        return estimated_nll.mean().neg().item()

    def _compute_vimco_baseline(self, ls):
        s = torch.sum(ls, dim=1, keepdim=True)
        repeated_ls = ls.unsqueeze(dim=1).repeat(1, self.num_samples, 1)
        prev_baseline = repeated_ls + (1. / (self.num_samples - 1)) * torch.diag_embed(s - self.num_samples * ls)
        baseline = torch.logsumexp(prev_baseline, dim=2)  # [bs, num_samples]
        return baseline

    def train_vimco(self, x):
        xx = x.unsqueeze(dim=1).repeat(1, self.num_samples, 1)
        log_pxh, log_qh_x = self.compute_logqh_x_and_logpxh(xx)
        ls = log_pxh - log_qh_x  # [bs, num_samples]
        elbo = torch.logsumexp(ls, dim=1, keepdim=True)
        baselines = self._compute_vimco_baseline(ls)
        ws = (ls - elbo).exp().detach()  # [bs, num_samples]
        learning_signal = (elbo - baselines).detach()  # [bs, num_samples]
        loss_for_theta = (ws * log_pxh).sum(dim=1).mean().neg()
        loss_for_phi = (learning_signal * log_qh_x - ws * log_qh_x).sum(dim=1).mean().neg()
        total_loss = loss_for_theta + loss_for_phi
        return total_loss, (elbo - math.log(self.num_samples)).mean().neg().item()

    def eval_multi_sample_elbo(self, x):
        xx = x.unsqueeze(dim=1).repeat(1, self.num_samples, 1)
        log_pxh, log_qh_x = self.compute_logqh_x_and_logpxh(xx)
        elbo = torch.logsumexp(log_pxh - log_qh_x, dim=1, keepdim=True) - math.log(self.num_samples)
        return elbo.mean().neg().item()

    def _compute_nvil_idbs(self, layer_wise_input):
        baseline = []
        for i, fnn in enumerate(self.idb):
            baseline.append(fnn(layer_wise_input[i]).squeeze())
        return torch.stack(baseline, dim=1)

    def train_nvil(self, x):
        log_prior = []  # p(h_1 | h_2), p(h_2 | h_3), ..., p(h_n)
        log_posterior = []  # q(h_1 | x), q(h_2 | h_1), q(h_3 | h_2) ..., q(h_n | h_n-1)
        logit_q, sampled_h = self.inference_net((x - self.mean_obs + 1.) / 2)
        # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_p_x_h = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        log_ll = log_bernoulli_prob(logit_p_x_h[-1], x)  # log likelihood
        logit_p = [self.top_prior(sampled_h[-1])] + logit_p_x_h[:-1]  # logit of [h_n , h_n-1 , ...,  h_1] for prior

        inputs = [x - self.mean_obs] + sampled_h
        idbs = self._compute_nvil_idbs(inputs)

        for i in range(self.num_layers):
            log_prior.append(log_bernoulli_prob(logit_p[-(i + 1)], sampled_h[i]))
            log_posterior.append(log_bernoulli_prob(logit_q[i], sampled_h[i]))
        log_pxh, log_qh_x = torch.stack([log_ll] + log_prior, dim=1), torch.stack(log_posterior, dim=1)

        ls_per_layer = (torch.matmul(self.tri_1, log_pxh.unsqueeze(2)) -
                        torch.matmul(self.tri_2, log_qh_x.unsqueeze(2))).squeeze(dim=2)
        ls_sub_idb = ls_per_layer - idbs
        with torch.no_grad():
            elbo = log_pxh.sum(dim=1) - log_qh_x.sum(dim=1)
            # assign less weight on the first batch
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * torch.mean(ls_sub_idb, dim=0,
                                                                                               keepdim=True)
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * torch.var(ls_sub_idb, dim=0,
                                                                                            keepdim=True)
            learning_signal = (ls_sub_idb - self.running_mean) / torch.max(self.all_ones, self.running_var.sqrt())
        loss_for_theta = log_pxh.sum(dim=1).mean().neg()
        loss_for_phi = (learning_signal * log_qh_x).sum(dim=1).mean().neg()
        loss_for_psi = (learning_signal * idbs).sum(dim=1).mean().neg()
        total_loss = loss_for_theta + loss_for_phi + loss_for_psi
        return total_loss, elbo.mean().neg().item()

    def eval_single_sample_elbo(self, x):
        log_pxh, log_qh_x = self.compute_logqh_x_and_logpxh(x)
        return (log_pxh - log_qh_x).mean().neg().item()

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

    def _compute_fxh(self, x, h, logit_q):
        logit_pxh = self.generative_net([h, x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_ph = [self.top_prior(h)] + logit_pxh[:-1]
        log_px_h = log_bernoulli_prob(logit_pxh[-1], x)
        log_qh_x = log_bernoulli_prob(logit_q, h)
        log_ph = log_bernoulli_prob(logit_ph[-1], h)
        return log_px_h + log_ph - log_qh_x

    def train_arm(self, x):
        logit_q, sampled_h = self.inference_net((x - self.mean_obs + 1.) / 2)
        # logit of [h_1 , h_2 , ... , h_n] for posterior
        logit_pxh = self.generative_net(sampled_h[::-1] + [x])  # logit of h_n-1, h_n-2, ..., h_1, x
        logit_p = [self.top_prior(sampled_h[-1])] + logit_pxh[:-1]
        log_px_h = log_bernoulli_prob(logit_pxh[-1], x)
        log_ph = log_bernoulli_prob(logit_p[-1], sampled_h[0])
        log_qh_x = log_bernoulli_prob(logit_q[0], sampled_h[0])
        with torch.no_grad():
            elbo = (log_ph + log_px_h - log_qh_x).mean().neg().item()
            u = torch.rand_like(logit_q[0])
            prob = torch.sigmoid(logit_q[0])
            z_1 = (u > (1. - prob)).float()
            z_2 = (u < prob).float()
            f_1 = self._compute_fxh(x, z_1, logit_q[0]).unsqueeze(dim=1)
            f_2 = self._compute_fxh(x, z_2, logit_q[0]).unsqueeze(dim=1)
        loss_for_theta = torch.sum((f_1 - f_2) * (u - 0.5) * logit_q[0], dim=-1)
        total_loss = (log_ph + log_px_h + loss_for_theta).mean().neg()
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
