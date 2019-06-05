import torch
import time
import sys
import math

from utils import AverageMeter, calculate_accuracy

from tensorboardX import SummaryWriter


def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    write_embedding = True
    writer = None
    embedding_log = 20

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            if not opt.no_cuda:
                targets = targets.cuda(async=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies))

            if write_embedding and epoch % embedding_log == 0:
                if writer is None:
                    writer = SummaryWriter(comment='_embedding_val_'+str(i))
                n_iter = (epoch * len(data_loader)) + i
                middle_frame = math.floor(inputs.data.shape[2] / 2)
                writer.add_embedding(
                    outputs.data,
                    metadata=targets.data,
                    label_img=torch.squeeze(
                        inputs.data[:, :, middle_frame, :, :], 2),
                    global_step=n_iter)

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

    return epoch, losses.avg, accuracies.avg
