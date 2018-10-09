import os
import torch
import models

def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out =  0.5*((logits-target)**2).sum()
    return out

def train_epoch(loader, model, criterion, weight_quantizer, grad_quantizer,
                writer, epoch, quant_bias=True, quant_bn=True, log_error=False,
                wage_quantize=False, wage_grad_clip=None):
    loss_sum = 0.0
    correct = 0.0
    semi_correct = 0.0

    model.train()
    ttl = 0

    for i, (input_v, target) in enumerate(loader):
        step = i+epoch*len(loader)
        input_v = input_v.cuda(async=True)
        # input is [0-1], scale to [-1,1]
        input_v = input_v*2-1
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_v)
        target_var = torch.autograd.Variable(target)

        # WAGE quantize 8-bits accumulation into ternary before forward
        # assume no batch norm
        for name, param in model.named_parameters():
            param.data = weight_quantizer(model.weight_acc[name], model.weight_scale[name])

        # Write ternary parameters
        if log_error:
            for name, param in model.named_parameters():
                writer.add_histogram("param-acc/%s"%name,
                    model.weight_acc[name].clone().cpu().data.numpy(), step)
                writer.add_histogram(
                    "param-quant/%s"%name, param.clone().cpu().data.numpy(), step)

        output = model(input_var)
        loss = criterion(output, target_var)

        if log_error:
            writer.add_scalar( "batch-train-loss", loss.item(), step)
            writer.add_histogram("output", output.cpu().data.numpy(), step)

        model.zero_grad()
        loss.backward()

        # Write high precision gradient
        if log_error:
            for name, param in model.named_parameters():
                writer.add_histogram(
                    "gradient-before/%s"%name, param.grad.clone().cpu().data.numpy(), step)

        # gradient quantization
        for name, param in list(model.named_parameters())[::-1]:
            param.grad.data = grad_quantizer(param.grad.data).data

            # Write 8-bits gradients
            if log_error:
                writer.add_histogram(
                    "gradient-after/%s"%name, param.grad.clone().cpu().data.numpy(), step)

            # WAGE accumulate weight in gradient precision
            # assume no batch norm
            w_acc =  wage_grad_clip(model.weight_acc[name])
            w_acc -= param.grad.data
            model.weight_acc[name] = w_acc

        loss_sum += loss.cpu().item() * input_v.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum()
        ttl += input_v.size()[0]

        # Compute in_top_k, similar to tensorflow
        max_output = output.max(1, keepdim=True)
        semi_correct += torch.eq(
            output[torch.arange(pred.size(0)), target],
            output.max(1)[0]
        ).sum()

    semi_correct = semi_correct.cpu().item()
    correct = correct.cpu().item()
    return {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
        'semi_accuracy': semi_correct / float(ttl) * 100.0,
    }


def eval(loader, model, criterion, wage_quantizer=None):
    loss_sum = 0.0
    correct = 0.0
    semi_correct = 0.0

    model.eval()
    cnt = 0

    # WAGE quantize 8-bits accumulation into ternary before forward
    # assume no batch norm
    for name, param in model.named_parameters():
        param.data = wage_quantizer(model.weight_acc[name], model.weight_scale[name])

    with torch.no_grad():
        for i, (input_v, target) in enumerate(loader):
            input_v = input_v.cuda(async=True)
            input_v = input_v*2-1
            target = target.cuda(async=True)

            output = model(input_v)
            loss = criterion(output, target)

            loss_sum += loss.data.cpu().item() * input_v.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            cnt += int(input_v.size()[0])

            # Compute in_top_k, similar to tensorflow
            max_output = output.max(1, keepdim=True)
            semi_correct += torch.eq(
                output[torch.arange(pred.size(0)), target],
                output.max(1)[0]
            ).sum()

    correct = correct.cpu().item()
    semi_correct = semi_correct.cpu().item()

    return {
        'loss': loss_sum / float(cnt),
        'accuracy': correct / float(cnt) * 100.0,
        'semi_accuracy': semi_correct / float(cnt) * 100.0,
    }

