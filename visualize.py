import numpy as np
import matplotlib.pyplot as plt
import re
from os import listdir
from os.path import isfile, join

plt.style.use('ggplot')


def alg_layer_and_samples(fn):
    print(fn)
    # pattern = r'(vimco-\d*_test)|(nvil_test)'
    pattern = r'(vimco-\d*_test)|(nvil_test)'
    alg_samples = re.search(pattern, fn)
    if alg_samples is None:
        return ''
    print(alg_samples.group() + '-' + re.search(r'\d\d\d(-\d\d\d)*', fn).group())
    return alg_samples.group() + '-' + re.search(r'\d\d\d(-\d\d\d)*', fn).group()


log_path = 'logs'
data_fn = [f for f in listdir(log_path) if isfile(join(log_path, f)) and f.endswith('.npy')]
print(data_fn)
all_data = [np.load(join(log_path, f)) for f in data_fn]
data_digest = [alg_layer_and_samples(f) for f in data_fn]

print(data_digest)
train_data = [
    np.load('logs/sbn/arm-nonlinear_train_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/legrad-nonlinear_train_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/nvil-nonlinear_train_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/rebar-nonlinear_train_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/gumbel-nonlinear_train_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/st-nonlinear_train_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy')
]

test_data = [
    np.load('logs/sbn/arm-nonlinear_test_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/legrad-nonlinear_test_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/nvil-nonlinear_test_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/rebar-nonlinear_test_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/gumbel-nonlinear_test_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy'),
    np.load('logs/sbn/st-nonlinear_test_bs24_dsstatic_mnist_layer200_genlr0.0001_inflr0.0001.npy')
]

## TODO last checkpoint : 2020/06/03 before 19:18. Go through local history
plt.plot(train_data[0], label='arm_train')
plt.plot(train_data[1], label='legrad_train')
plt.plot(train_data[2], label='nvil_train')
plt.plot(train_data[3], label='rebar_train')
plt.plot(train_data[4], label='gumbel_train')
plt.plot(train_data[5], label='st_train')
plt.legend()
plt.ylim(95, 160)
plt.ylabel("negative ll")
# plt.xticks([index + 0.1 for index in x], label_list)
plt.xlabel("epochs")
plt.title("Training Performance of different algs.")
plt.savefig('logs/plot_train.pdf')

plt.clf()
plt.plot(test_data[0], label='arm_test')
plt.plot(test_data[1], label='legrad_test')
plt.plot(test_data[2], label='nvil_test')
plt.plot(test_data[3], label='rebar_test')
plt.plot(test_data[4], label='gumbel_test')
plt.plot(test_data[5], label='st_test')
plt.legend()
plt.ylim(95, 140)
plt.ylabel("negative ll")
# plt.xticks([index + 0.1 for index in x], label_list)
plt.xlabel("epochs")
plt.title("Testing Performance of different algs.")
plt.savefig('logs/plot_test.pdf')
