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

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True,
                    help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10',
                    help='dataset name: CIFAR10 or IMAGENET12')
parser.add_argument('--data_path', type=str, default="./data", required=True, metavar='PATH',
                    help='path to datasets location (default: "./data")')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--seed', type=int, default=200, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-name', type=str, default='', metavar='S',
                    help="Name for the log dir")
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
                    help='word length in bits for rand number; -1 if full precision.')

args = parser.parse_args()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)


# Tensorboard Writer
if args.log_name != "":
    log_name = args.log_name
else:
    log_name = "time%d"%int(time.time)
print("Logging at {}".format(log_name))
writer = SummaryWriter(log_dir=os.path.join(".", "runs", log_name))


# Select quantizer
def quant_summary(wl, fl):
    if wl == -1:
        return "float"
    else:
        return "wage-{}".format(wl)

w_summary = quant_summary(args.wl_weight, args.fl_weight)
g_summary = quant_summary(args.wl_grad, args.fl_grad)
a_summary = quant_summary(args.wl_activate, args.fl_activate)
e_summary = quant_summary(args.wl_error, args.fl_error)
print("W:{}, A:{}, G:{}, E:{}".format(w_summary, a_summary, g_summary, e_summary))

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

if args.dataset=="CIFAR10":
    ds = getattr(datasets, args.dataset)
    path = os.path.join(args.data_path, args.dataset.lower())
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = ds(path, train=True, download=True, transform=transform_train)
    val_set = ds(path, train=True, download=True, transform=transform_test)
    test_set = ds(path, train=False, download=True, transform=transform_test)
    train_sampler = None
    val_sampler = None
    num_classes = 10

loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'val': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
}

# Build model
print('Model: {}'.format(args.model))
model_cfg = getattr(models, args.model)
model_cfg.kwargs.update({
    "wl_activate":args.wl_activate, "fl_activate":args.fl_activate,
    "wl_error":args.wl_error, "fl_error":args.fl_error,
    "wl_weight":args.wl_weight,
})

if args.log_error:
    model = model_cfg.base(
            *model_cfg.args,
            num_classes=num_classes, writer=writer,
            **model_cfg.kwargs)
else:
    model = model_cfg.base(
            *model_cfg.args,
            num_classes=num_classes, writer=None,
            **model_cfg.kwargs)
model.cuda()
for name, param_acc in model.weight_acc.items():
    model.weight_acc[name] = param_acc.cuda()

criterion = utils.SSE

def schedule(epoch):
    if epoch < 200:
        return 8.0
    elif epoch < 250:
        return 1
    else:
        return 1/8.

start_epoch = 0

# Prepare logging
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'tr_acc2',
           'te_loss', 'te_acc', 'te_acc2', 'time']

def log_result(writer, name, res, step):
    writer.add_scalar("{}/loss".format(name),     res['loss'],            step)
    writer.add_scalar("{}/acc_perc".format(name), res['accuracy'],        step)
    writer.add_scalar("{}/err_perc".format(name), 100. - res['accuracy'], step)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    lr = schedule(epoch)
    writer.add_scalar("lr", lr, epoch)
    grad_quantizer = lambda x : models.QG(x, args.wl_grad, args.wl_rand, lr)

    train_res = utils.train_epoch(
            loaders['train'], model, criterion,
            weight_quantizer, grad_quantizer, writer, epoch,
            log_error=args.log_error,
            wage_quantize=True,
            wage_grad_clip=grad_clip
    )
    log_result(writer, "train", train_res, epoch+1)

    # Validation
    test_res = utils.eval(loaders['test'], model, criterion, weight_quantizer)
    log_result(writer, "test", test_res, epoch+1)

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'],
            train_res['semi_accuracy'], test_res['loss'], test_res['accuracy'],
            test_res['semi_accuracy'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 20 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


