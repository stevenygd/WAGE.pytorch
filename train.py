import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import utils
import tabulate
import models
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
import json
from data_loaders import get_data_loaders

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True,
                    help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name: CIFAR10 or IMAGENET12')
parser.add_argument('--data_path', type=str, default="./data", required=True, metavar='PATH',
                    help='path to datasets location (default: "./data")')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--val_ratio', type=float, default=0.0, metavar='N',
                    help='Ratio of the validation set (default: 0.0)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N',
                    help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                    help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=8.0, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--lr-type', type=str, default="wilson", metavar='S',
                    choices=["wilson", "gupta", "const", "wage"],
                    help='learning decay schedule type ("wilson" or "gupta" or "const" or "wage")')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--swa', action='store_true',
                    help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N',
                    help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=1./8., metavar='N',
                    help='SWA learning rate (default: 1/8)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--c_ratio', type=float, default=1, metavar='N',
                    help='average per ratio * epoch')
parser.add_argument('--seed', type=int, default=200, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-name', type=str, default='', metavar='S',
                    help="Name for the log dir")
parser.add_argument('--log-distribution', action='store_true', default=False,
                    help='Whether to log distribution of weight and grad')
parser.add_argument('--log-error', action='store_true', default=False,
                    help='Whether to log quantization error of weight and grad')
parser.add_argument('--wl-weight', type=int, default=-1, metavar='N',
                    help='word length in bits for weight output; -1 if full precision.')
parser.add_argument('--fl-weight', type=int, default=-1, metavar='N',
                    help='float length in bits for weight output; -1 if full precision.')
parser.add_argument('--wl-grad', type=int, default=-1, metavar='N',
                    help='word length in bits for gradient; -1 if full precision.')
parser.add_argument('--fl-grad', type=int, default=-1, metavar='N',
                    help='float length in bits for gradient; -1 if full precision.')
parser.add_argument('--wl-activate', type=int, default=-1, metavar='N',
                    help='word length in bits for layer activattions; -1 if full precision.')
parser.add_argument('--fl-activate', type=int, default=-1, metavar='N',
                    help='float length in bits for layer activations; -1 if full precision.')
parser.add_argument('--wl-error', type=int, default=-1, metavar='N',
                    help='word length in bits for backward error; -1 if full precision.')
parser.add_argument('--fl-error', type=int, default=-1, metavar='N',
                    help='float length in bits for backward error; -1 if full precision.')
parser.add_argument('--wl-rand', type=int, default=-1, metavar='N',
                    help='word length in bits for rand number (WAGE replication only); \
                    -1 if full precision.')
parser.add_argument('--weight-type', type=str, default="fixed", metavar='S',
                    choices=["fixed", "block", "wage"],
                    help='quantization type for weight; fixed or block.')
parser.add_argument('--grad-type', type=str, default="fixed", metavar='S',
                    choices=["fixed", "block", "wage"],
                    help='quantization type for gradient; fixed or block.')
parser.add_argument('--layer-type', type=str, default="fixed", metavar='S',
                    choices=["fixed", "block", "wage"],
                    help='quantization type for layer activation and error; fixed or block.')
parser.add_argument('--quant-type', type=str, default='stochastic', metavar='S',
                    choices=["stochastic", "nearest"],
                    help='rounding method, stochastic or nearest ')

# Leraning rate schedule
parser.add_argument('--lr_changes', type=int,  nargs='+', help='Epoches when to change the learning rate',
                    default=[200,250,300])
parser.add_argument('--lr_schedules', type=float, nargs='+', help='<Required> Set flag',
                    default=[8,1,0.125])


args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

# Tensorboard Writer
if args.log_name != "":
    log_name = "{}-{}-seed{}-{}".format( args.log_name, "swa" if args.swa else "sgd", args.seed, int(time.time()))
else:
    log_name = "{}-seed{}-{}".format("swa" if args.swa else "sgd", args.seed, int(time.time()))
print("Logging at {}".format(log_name))
writer = SummaryWriter(log_dir=os.path.join(".", "runs", log_name))


# Select quantizer
def quant_summary(number_type, wl, fl):
    if wl == -1:
        return "float"
    if number_type=="fixed":
        return "fixed-{}{}".format(wl, fl)
    elif number_type=="block":
        return "block-{}".format(wl)
    elif number_type=="wage":
        return "wage-{}".format(wl)

w_summary = quant_summary(args.weight_type, args.wl_weight, args.fl_weight)
g_summary = quant_summary(args.grad_type, args.wl_grad, args.fl_grad)
a_summary = quant_summary(args.layer_type, args.wl_activate, args.fl_activate)
e_summary = quant_summary(args.layer_type, args.wl_error, args.fl_error)
print("{} rounding, W:{}, A:{}, G:{}, E:{}".format(args.quant_type, w_summary,
                                                   a_summary, g_summary,
                                                   e_summary))

assert args.weight_type == "wage" and args.grad_type == "wage"
weight_quantizer = lambda x, scale: models.QW(x, args.wl_weight, scale)
grad_clip = lambda x : models.C(x, args.wl_weight)
if args.wl_weight==-1: weight_quantizer = None
if args.wl_grad ==-1: grad_quantier = None


dir_name = args.dir + "-seed-{}".format(args.seed)
print('Preparing checkpoint directory {}'.format(dir_name))
try:
    os.makedirs(dir_name)
except:
    pass
with open(os.path.join(dir_name, 'command.sh'), 'w') as f:
    f.write('python '+' '.join(sys.argv))
    f.write('\n')

assert args.dataset in ["CIFAR10"]
print('Loading dataset {} from {}'.format(args.dataset, args.data_path))

# Build model
print('Model: {}'.format(args.model))
model_cfg = getattr(models, args.model)
if 'WAGE' in args.model and (args.wl_activate != -1 or args.wl_error != -1):
    model_cfg.kwargs.update(
        {"wl_activate":args.wl_activate, "fl_activate":args.fl_activate,
         "wl_error":args.wl_error, "fl_error":args.fl_error,
         "wl_weight":args.wl_weight,
         "layer_type":args.layer_type})

model_writer = writer if args.log_error else None
num_classes = 10
model = model_cfg.base(
    *model_cfg.args,
    num_classes=num_classes, writer=model_writer,
    **model_cfg.kwargs)
model.cuda()
# hook momentum buffer to the model
if args.momentum != 0:
    model.momentum_buffer = {}
for name, param_acc in model.weight_acc.items():
    model.weight_acc[name] = param_acc.cuda()

if args.swa:
    print('SWA training')
    ratios = [0.1, 0.05, 0.025, 0.01, 0.005]
    cratio_swa_model_dict = {}
    for r in ratios:
       swa_model_dict = {}
       model_names = ['full_acc', 'low_acc', 'full_tern', 'low_tern']
       for model_name in model_names:
           # use the same model config, i.e., quantizing activations
           swa_model = model_cfg.base(
               *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
           swa_model.cuda()
           for name, param_acc in swa_model.weight_acc.items():
               swa_model.weight_acc[name] = param_acc.cuda()
           swa_model.weight_scale = model.weight_scale
           swa_model_dict[model_name] = swa_model
           swa_model_dict['swa_n'] = 0
       cratio_swa_model_dict[r] = swa_model_dict

def swa_quantizer(weight, scale, threshold):
    mask = weight.abs() > threshold
    sign = weight.sign()
    quantized = weight.clone()
    quantized.masked_fill_(mask, 0.5)
    quantized.masked_fill_(1-mask, 0)
    quantized = quantized * sign
    quantized = quantized / scale
    return quantized

assert args.weight_type == "wage"
criterion = utils.SSE

def schedule(epoch):
    lr = args.lr_init
    for e,l in zip(args.lr_changes, args.lr_schedules):
        if epoch >= e:
            lr = l
    if args.swa and (epoch+1) > args.swa_start:
        lr =  args.swa_lr
    return lr

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']-1
    model.weight_acc = checkpoint['acc_dict']


# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_acc2',
           'te_loss', 'te_acc', 'te_acc2', 'time']
if args.swa:
    columns = columns[:-1] + ['swa_te_loss', 'swa_te_acc'] + columns[-1:]
    swa_res = {'loss': None, 'accuracy': None}

loaders = get_data_loaders(args.dataset, args.data_path, args.val_ratio, args.batch_size, args.num_workers)

# Save SWA models
utils.save_checkpoint(
    dir_name,
    0,
    acc_dict=model.weight_acc,
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    lr = schedule(epoch)
    writer.add_scalar("lr", lr, epoch)
    assert args.grad_type == 'wage'
    grad_quantizer = lambda x : models.QG(x, args.wl_grad, args.wl_rand, lr)


    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        train_res = utils.train_epoch(
            loaders, model, criterion,
            weight_quantizer, grad_quantizer, writer, epoch,
            log_error=args.log_error,
            wage_quantize=True,
            wage_grad_clip=grad_clip,
            momentum = args.momentum,
            cratio_swa_model_dict = cratio_swa_model_dict,
        )
    else:
        train_res = utils.train_epoch(
            loaders, model, criterion,
            weight_quantizer, grad_quantizer, writer, epoch,
            log_error=args.log_error,
            wage_quantize=True,
            wage_grad_clip=grad_clip,
            momentum = args.momentum,
            cratio_swa_model_dict = None
           )
    utils.log_result(writer, "train", train_res, epoch+1)


    all_result = {}
    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        for ratio, swa_model_dict in cratio_swa_model_dict.items():
            for swa_model_name, swa_model in swa_model_dict.items():
                if 'full' in swa_model_name:
                    test_res = utils.eval(loaders['test'], swa_model, criterion, None)
                    utils.log_result(writer, 'c{}/{}_test'.format(ratio, swa_model_name), test_res, epoch+1)
                elif 'low' in swa_model_name:
                    test_res = utils.eval(loaders['test'], swa_model, criterion, weight_quantizer)
                    utils.log_result(writer, 'c{}/{}_test'.format(ratio, swa_model_name), test_res, epoch+1)
                elif swa_model_name == 'swa_n': continue
                else: raise ValueError("invalid swa model name {}".format(swa_model_name))
                all_result["c{}/{}_test".format(ratio, swa_model_name)] = test_res

    # Write parameters
    if args.log_distribution:
        for name, param in model.named_parameters():
            writer.add_histogram(
                "param/%s"%name, param.clone().cpu().data.numpy(), epoch)
            writer.add_histogram(
                "param/wacc-%s"%name,
                model.weight_acc[name].clone().cpu().data.numpy(), epoch)
            writer.add_histogram(
                "gradient/%s"%name, param.grad.clone().cpu().data.numpy(), epoch)


    # Validation
    test_res = utils.eval(loaders['test'], model, criterion, None)
    utils.log_result(writer, "base_acc_test", test_res, epoch+1)
    all_result["base_acc_test"] = test_res
    test_res = utils.eval(loaders['test'], model, criterion, weight_quantizer)
    utils.log_result(writer, "base_test", test_res, epoch+1)
    all_result["base_test"] = test_res

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'],
            train_res['semi_accuracy'], test_res['loss'], test_res['accuracy'],
            test_res['semi_accuracy'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

    if (epoch+1) % args.save_freq == 0:
        utils.save_checkpoint(
            dir_name,
            epoch + 1,
            acc_dict=model.weight_acc,
        )

os.makedirs("diffc", exist_ok=True)
ratio_result = {}
for avg_name in ['full_acc', 'low_acc', 'full_tern', 'low_tern']:
    ratio_result[avg_name] = []
for ratio, swa_model_dict in cratio_swa_model_dict.items():
    for swa_model_name, swa_model in swa_model_dict.items():
        if swa_model_name == "swa_n": continue
        swa_model = swa_model_dict[swa_model_name]
        if 'full' in swa_model_name:
            test_res = utils.eval(loaders['test'], swa_model, criterion, None)
        elif 'low' in swa_model_name:
            test_res = utils.eval(loaders['test'], swa_model, criterion, weight_quantizer)
        ratio_result[swa_model_name].append(100-test_res['accuracy'])
for avg_name in ['full_acc', 'low_acc', 'full_tern', 'low_tern']:
    with open("diffc/{}_{}_result.csv".format(args.log_name, avg_name), "a") as f:
        f.write("{},".format(args.seed))
        f.write(",".join([str(acc) for acc in ratio_result[avg_name]]))
        f.write("\n")
