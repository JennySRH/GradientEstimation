import torch
import argparse
import load_dataset
from sbn import SigmoidBeliefNetwork
from torch.optim import Adam, SGD
from load_dataset import load_dataset
from utils import MultiOptim
import numpy as np

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--test-batch-size', type=int, default=512)
parser.add_argument('-e', '--epochs', type=int, default=1000)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--prior-lr', type=float, default=1e-2)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--dims', type=int, nargs='+', default=[200, 300, 400],
                    help='Specify number of hidden layers '
                         'and dimensions within each layer.')
parser.add_argument('-d', '--dataset', type=str, default='static-mnist')
parser.add_argument('-m', '--model', type=str, default='nvil')
parser.add_argument('-k', '--num-samples', type=int, default=5)
parser.add_argument('--test-samples', type=int, default=1000)
parser.add_argument('--use-output-bias', action='store_true', default=False)
parser.add_argument('--use-argen', action='store_true', default=False)
parser.add_argument('--use-arinf', action='store_true', default=False)
parser.add_argument('--use-nonlinear', action='store_true', default=False)
parser.add_argument('--use-uniform', action='store_true', default=False)
parser.add_argument("--gpu-num", type=str, default="0", help="gpu number")
args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
args.cuda = torch.cuda.is_available()
print('{}'.format(args))
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train_sbn(train_loader, model, optim, args):
    model.train()
    train_elbo = 0.
    for batch_idx, (data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        optim.zero_grad()
        loss, elbo = model.train_step(data)
        loss.backward()
        train_elbo += elbo
        optim.step()
    train_elbo /= len(train_loader)
    return train_elbo


def eval_sbn(test_loader, model, args):
    model.eval()
    eval_elbo = 0.
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            if args.cuda:
                data = data.cuda()
            elbo = model.eval_step(data)
            eval_elbo += elbo.mean().neg().item()
    eval_elbo /= len(test_loader)
    return eval_elbo


def calc_nll(test_loader, model, args):
    model.eval()
    num_samples = args.test_samples
    with torch.no_grad():
        test_ll = 0.
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(test_loader):
                if args.cuda:
                    data = data.cuda()
                log_ll = model.compute_multisample_bound(data, num_samples).mean().neg().item()
                test_ll += log_ll
        test_ll /= len(test_loader)
        return test_ll


def run():
    train_loader, val_loader, test_loader, mean_obs = load_dataset(args)
    if args.use_output_bias:
        init_bias = -torch.log(1./torch.clamp(mean_obs, 0.001, 0.999) - 1.)
    else:
        init_bias = None
    train_elbos = []
    eval_elbos = []
    model = SigmoidBeliefNetwork(mean_obs, init_bias,
                dim_hids=args.dims,
                use_nonlinear=args.use_nonlinear,
                use_ar_gen=args.use_argen,
                use_ar_inf=args.use_arinf,
                use_uniform_prior=args.use_uniform,
                method=args.model,
                num_samples=args.num_samples,
                temp=args.temperature
                )

    if args.model == 'nvil':
        param_list = list(model.generative_net.parameters()) + list(model.inference_net.parameters()) + list(model.idb.parameters())
    else:
        param_list = list(model.generative_net.parameters()) + list(model.inference_net.parameters())
    print(param_list)
    # according to https://github.com/mingzhang-yin/ARM-gradient/blob/master/b_omni_linear.py#L192,
    # the top prior parameters p(b_L) are optimized using SGD with learning rate 0.01;
    # and all the other parameters are optimized using Adam with learning rate 0.0001.
    optim_1 = Adam(param_list, lr=args.lr)
    optim_2 = SGD(model.top_prior.parameters(), lr=args.prior_lr)
    optim = MultiOptim(optim_1, optim_2)
    if args.model == 'vimco':
        args.model_config = args.model + '-' + str(args.num_samples)
    else:
        args.model_config = args.model
    if args.use_nonlinear:
        args.model_config += '-nonlinear'
    else:
        if args.use_argen:
            args.model_config += '-argen'
        if args.use_arinf:
            args.model_config += '-arinf'
    if args.cuda:
        model.cuda()
    for epoch in range(args.epochs):
        train_elbo = train_sbn(train_loader, model, optim, args)
        eval_elbo = eval_sbn(test_loader, model, args)
        train_elbos.append(train_elbo)
        eval_elbos.append(eval_elbo)
        print('Epoch: {}/{}, * Train loss: {:.4f}, o Eval loss: {:.4f}'.format(
            epoch + 1, args.epochs, train_elbo, eval_elbo
        ))
    test_ll = calc_nll(test_loader, model, args)
    print('Test loss : {:.4f}'.format(test_ll))
    np.save('logs/sbn/{}_train_bs{}_ds{}_layer{}_genlr{}_inflr{}.npy'.format(
        args.model_config,
        args.batch_size,
        args.dataset,
        '-'.join(str(dim) for dim in args.dims),
        args.gen_lr,
        args.inf_lr
    ), np.array(train_elbos))
    np.save('logs/sbn/{}_test_bs{}_ds{}_layer{}_genlr{}_inflr{}.npy'.format(
        args.model_config,
        args.batch_size,
        args.dataset,
        '-'.join(str(dim) for dim in args.dims),
        args.gen_lr,
        args.inf_lr
    ), np.array(eval_elbos))
    with open('logs/sbn/ex.txt', 'a+') as f:
        f.write('config: {} \ntrain loss : {:.4f}, eval loss : {:.4f}, test ll : {:.4f}\n'.
                format(args, train_elbos[-1], eval_elbos[-1], test_ll))


if __name__ == "__main__":
    run()
