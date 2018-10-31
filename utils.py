import os
import torch
import models

def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out =  0.5*((logits-target)**2).sum()
    return out

def save_checkpoint(dir, epoch, **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)

def log_result(writer, name, res, step):
    writer.add_scalar("{}/loss".format(name),     res['loss'],            step)
    writer.add_scalar("{}/acc_perc".format(name), res['accuracy'],        step)
    writer.add_scalar("{}/err_perc".format(name), 100. - res['accuracy'], step)

def train_epoch(loaders, model, criterion, weight_quantizer, grad_quantizer,
                writer, epoch, quant_bias=True, quant_bn=True, log_error=False,
                wage_quantize=False, wage_grad_clip=None, momentum=0,
                cratio_swa_model_dict=None):
    loss_sum = 0.0
    correct = 0.0
    semi_correct = 0.0

    model.train()
    ttl = 0

    for i, (input_v, target) in enumerate(loaders['train']):
        step = i+epoch*len(loaders['train'])
        input_v = input_v.cuda()
        target = target.cuda()
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
            # param.grad.data = grad_quantizer(param.grad.data).data

            if momentum != 0:
                if not name in model.momentum_buffer:
                    buf = model.momentum_buffer[name] = torch.zeros_like(param.data)
                    buf.mul_(momentum).add_(param.grad.data)
                else:
                    buf = model.momentum_buffer[name]
                    buf.mul_(momentum).add_(param.grad.data)

                model.momentum_buffer[name] = grad_quantizer(buf.data.clone())
                param.grad.data = model.momentum_buffer[name].clone()
            else:
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

        max_output = output.max(1, keepdim=True)
        semi_correct += torch.eq(
            output[torch.arange(pred.size(0)), target],
            output.max(1)[0]
        ).sum()

        if cratio_swa_model_dict != None:
            for ratio, swa_model_dict in cratio_swa_model_dict.items():
                if ((i+1) % int(ratio*len(loaders['train']))) == 0:
                    swa_n = swa_model_dict['swa_n']
                    decay = 1.0 / (swa_n+1)
                    for swa_model_name, swa_model in swa_model_dict.items():
                        if 'tern' in swa_model_name:
                            target = 'tern'
                        elif 'acc' in swa_model_name:
                            target = 'acc'
                        elif swa_model_name == 'swa_n': continue
                        else: raise ValueError("invalid swa model name {}".format(swa_model_name))
                        moving_average(swa_model, model, decay,
                                       average_target=target, swa_wl_weight=swa_model.wl_weight)
                    swa_n += 1
                    swa_model_dict['swa_n'] = swa_n

    semi_correct = semi_correct.cpu().item()
    correct = correct.cpu().item()
    return {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl) * 100.0,
        'semi_accuracy': semi_correct / float(ttl) * 100.0,
    }

def moving_average(swa_model, base_model, alpha=1, average_target="acc", swa_wl_weight=2):
    for name, _ in base_model.named_parameters():
        swa_acc = swa_model.weight_acc[name].data
        swa_acc *= (1.0-alpha)
        if average_target == "acc":
            w_acc = models.C(base_model.weight_acc[name].data, base_model.wl_weight)
            swa_acc += w_acc * alpha
        elif average_target == 'tern':
            swa_acc += models.QW(base_model.weight_acc[name], swa_wl_weight, scale=1.0) * alpha # not applying constant scaling when averaging
        else: raise ValueError("invalid target {}".format(average_target))
        swa_model.weight_acc[name] = swa_acc

def eval(loader, model, criterion, weight_quantizer=None):
    loss_sum = 0.0
    correct = 0.0
    semi_correct = 0.0

    model.eval()
    cnt = 0

    with torch.no_grad():
        # WAGE quantize 8-bits accumulation into ternary before forward
        # assume no batch norm
        for name, param in model.named_parameters():
            if weight_quantizer != None:
                param.data = weight_quantizer(model.weight_acc[name], model.weight_scale[name])
            else:
                param.data = model.weight_acc[name]/model.weight_scale[name] # apply constant scaling to full precision model

        for i, (input_v, target) in enumerate(loader):
            input_v = input_v.cuda()
            target = target.cuda()

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
