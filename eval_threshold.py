import argparse
import torch
import utils
import models
import numpy as np
from data_loaders import get_data_loaders
import matplotlib.pyplot as plt
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name: CIFAR10 or IMAGENET12')
parser.add_argument('--data_path', type=str, default="./data", required=False, metavar='PATH',
                    help='path to datasets location (default: "./data")')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--val_ratio', type=float, default=0.0, metavar='N',
                    help='Ratio of the validation set (default: 0.0)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default="WAGEVGG7", required=False, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default="./checkpoint/wage-replicate/swa210_lr8lr2-seed-{}/checkpoint-300.pt", metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--wl-weight', type=int, default=2, metavar='N',
                    help='word length in bits for weight output; -1 if full precision.')
parser.add_argument('--wl-activate', type=int, default=8, metavar='N',
                    help='word length in bits for layer activattions; -1 if full precision.')

args = parser.parse_args()

model_cfg = getattr(models, args.model)
model_cfg.kwargs.update(
    {"wl_activate":args.wl_activate, "wl_weight":args.wl_weight,
     "layer_type":"wage"})
num_classes = 10

model_dict = {}
for model_name in ["full_tern", "low_tern", "low_acc"]:
    model_dict[model_name] = []
    for seed in range(100, 200, 100):
        model = model_cfg.base(
            *model_cfg.args,
            num_classes=num_classes, writer=None,
            **model_cfg.kwargs).to('cuda')
        checkpoint = torch.load(args.resume.format(seed))
        model.weight_acc = checkpoint[model_name]
        model_dict[seed] = model
        model_dict[model_name].append(model)

loaders = get_data_loaders(args.dataset, args.data_path, args.val_ratio, args.batch_size, args.num_workers)

def swa_quantizer(weight, scale, threshold):
    mask = weight.abs() > threshold
    sign = weight.sign()
    quantized = weight.clone()
    quantized.masked_fill_(mask, 0.5)
    quantized.masked_fill_(1-mask, 0)
    quantized = quantized * sign
    quantized = quantized / scale
    return quantized

criterion = utils.SSE

for model in model_dict["full_tern"]:
    accs = []
    for model in model_dict["full_tern"]:
        accs.append(utils.eval(loaders['test'], model, criterion, None)["accuracy"])
    full_acc = sum(accs) / len(accs)

step = np.arange(0.24, 0.3, 1e-2)
print(step)

result_dict = {}
for name in ["low_acc", "low_tern"]:
    result_dict[name] = []
    for threshold in step:
        threshold = threshold.item()
        accs = []
        for model in model_dict[name]:
            threshold_quantizer = lambda weight, scale: swa_quantizer(weight, scale, threshold)
            accs.append(utils.eval(loaders['test'], model, criterion, threshold_quantizer)["accuracy"])
        acc = sum(accs) / len(accs)
        result_dict[name].append(acc)

fig = plt.figure()
plt.plot(step, result_dict["low_acc"])
plt.plot(step, result_dict["low_tern"])
plt.plot(step, [full_acc for _ in step], step)
fig.savefig("threshold_experiment.png")
