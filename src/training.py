from tqdm import tqdm
import time
import numpy as np
import torch

from eval.evaluate import AverageMeter, accuracy
from utilities import rand_bbox


def train_fn(model, optimizer, criterion, loader, device, cutmix_prob, beta):
    """
  Training method
  :param model: model to train
  :param optimizer: optimization algorithm
  :param criterion: loss function
  :param loader: data loader for either training or testing set
  :param device: torch device
  :return: (accuracy, loss) on the data
  """
    time_begin = time.time()
    score = AverageMeter()
    losses = AverageMeter()
    model.train()
    time_train = 0

    t = tqdm(loader)
    for i, (images, labels) in enumerate(t):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # CutMix implementation from https://github.com/clovaai/CutMix-PyTorch/blob/master/train.py
        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(images.size()[0])
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            # compute output
            logits = model(images)
            loss = criterion(logits, target_a) * lam + criterion(logits, target_b) * (1. - lam)
        else:
            # compute output
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        acc = accuracy(logits, labels)
        n = images.size(0)
        losses.update(loss.item(), n)
        score.update(acc.item(), n)

        t.set_description('(=> Training) Loss: {:.4f}'.format(losses.avg))

    time_train += time.time() - time_begin
    print('training time: ' + str(time_train))
    return score.avg, losses.avg
