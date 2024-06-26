import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from hausdorff import hausdorff_distance

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)

class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        # print("dices:", dices)
        return np.asarray(dices)

class HausdorffAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.class_num, dtype='float64')
        self.avg = np.asarray([0]*self.class_num, dtype='float64')
        self.sum = np.asarray([0]*self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = HausdorffAverage.get_hausdorff(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_hausdorff(logits, targets):
        logits = logits.cpu()
        targets = targets.cpu()
        logits = logits.detach().numpy()
        targets = targets.detach().numpy()
        hausdorff_list = []
        for class_index in range(targets.shape[1]):
            res = []
            for b in range(targets.shape[0]):
                for Z in range(targets.shape[2]):
                    # print(Z)
                    X = logits[b, class_index, Z, :, :]
                    # print(X.shape)
                    Y = targets[b, class_index, Z, :, :]
                    hausdorff = hausdorff_distance(X, Y, distance="euclidean")
                    res.append(hausdorff)
            hausdorff_mean = sum(res)/(targets.shape[2] * targets.shape[0])
            hausdorff_list.append(hausdorff_mean)
        # print(hausdorff_list)
        return np.asarray(hausdorff_list)
