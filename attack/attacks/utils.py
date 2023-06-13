import torch
import torch.nn as nn
import numpy as np
from collections import Counter


class SEC4SR_CrossEntropy(nn.CrossEntropyLoss):  # deal with something special on top of CrossEntropyLoss

    def __init__(self, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean', task='CSI'):
        super().__init__(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce,
                         reduction=reduction)

        assert task == 'CSI'  # CrossEntropy only supports CSI task

    def forward(self, scores, label):

        _, num_class = scores.shape
        device = scores.device
        label = label.to(device)
        loss = torch.zeros(label.shape[0], dtype=torch.float, device=scores.device)

        consider_index = torch.nonzero(label != -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
        if len(consider_index) > 0:
            loss[consider_index] = super().forward(scores[consider_index], label[consider_index])

        imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
        if len(imposter_index):
            loss[imposter_index] = 0. * torch.sum(scores[imposter_index])  # make backward

        return loss


class SEC4SR_MarginLoss(nn.Module):  # deal with something special on top of MarginLoss

    def __init__(self, targeted=False, confidence=0., task='CSI', threshold=None, clip_max=True) -> None:
        super().__init__()
        self.targeted = targeted
        self.confidence = confidence
        self.task = task
        self.threshold = threshold
        self.clip_max = clip_max

    def forward(self, scores, label):
        if len(scores.size()) == 1:
            scores = scores.unsqueeze(1)
        _, num_class = scores.shape
        device = scores.device
        label = label.to(device)
        loss = torch.zeros(label.shape[0], dtype=torch.float, device=scores.device)
        confidence = torch.tensor(self.confidence, dtype=torch.float, device=device)
        if self.task == 'SV':
            enroll_index = torch.nonzero(label == 1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            imposter_index = torch.nonzero(label == 0, as_tuple=True)[0].detach().cpu().numpy().tolist()
            assert len(enroll_index) + len(imposter_index) == label.shape[
                0], 'SV task should not have labels out of 0 and -1'
            if len(enroll_index) > 0:
                if self.targeted:
                    loss[enroll_index] = self.threshold + confidence - scores[enroll_index].squeeze(
                        1)  # imposter --> enroll, authentication bypass
                else:
                    loss[enroll_index] = scores[enroll_index].squeeze(
                        1) + confidence - self.threshold  # enroll --> imposter, Denial of Service
            if len(imposter_index) > 0:
                if self.targeted:
                    # enroll --> imposter, Denial of Service
                    loss[imposter_index] = scores[imposter_index].squeeze(1) + confidence - self.threshold
                else:
                    # imposter --> enroll, authentication bypass
                    loss[imposter_index] = self.threshold + confidence - scores[imposter_index].squeeze(1)

        elif self.task == 'CSI' or self.task == 'OSI':
            # remove imposter index which is unmeaningful for CSI task
            consider_index = torch.nonzero(label != -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            if len(consider_index) > 0:
                label_one_hot = torch.zeros((len(consider_index), num_class), dtype=torch.float, device=device)
                for i, ii in enumerate(consider_index):
                    index = int(label[ii])
                    label_one_hot[i][index] = 1
                score_real = torch.sum(label_one_hot * scores[consider_index], dim=1)
                score_other = torch.max((1 - label_one_hot) * scores[consider_index] - label_one_hot * 10000, dim=1)[0]
                if self.targeted:
                    loss[consider_index] = score_other + confidence - score_real if self.task == 'CSI' \
                        else torch.clamp(score_other, min=self.threshold) + confidence - score_real
                else:
                    if self.task == 'CSI':
                        loss[consider_index] = score_real + confidence - score_other
                    else:
                        f_reject = torch.max(scores[consider_index], 1)[
                                       0] + confidence - self.threshold  # spk m --> reject
                        f_mis = torch.clamp(score_real,
                                            min=self.threshold) + confidence - score_other  # spk_m --> spk_n
                        loss[consider_index] = torch.minimum(f_reject, f_mis)

            imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
            if self.task == 'OSI':
                # imposter_index = torch.nonzero(label == -1, as_tuple=True)[0].detach().cpu().numpy().tolist()
                if len(imposter_index) > 0:
                    if self.targeted:
                        loss[imposter_index] = torch.max(scores[imposter_index], 1)[0] + confidence - self.threshold
                    else:
                        loss[imposter_index] = self.threshold + confidence - torch.max(scores[imposter_index], 1)[0]
            else:  # CSI
                if len(imposter_index):
                    loss[imposter_index] = 0. * torch.sum(scores[imposter_index])  # make backward

            # else:
            #     loss[imposter_index] = torch.zeros(len(imposter_index))

        if self.clip_max:
            loss = torch.max(torch.tensor(0, dtype=torch.float, device=device), loss)

        return loss


class SpeakerVerificationLoss(nn.Module):
    def __init__(self, targeted=False, threshold=None):
        super().__init__()
        self.targeted = targeted
        self.threshold = threshold

    def forward(self, scores, label):
        device = scores.device
        label = label.to(device)
        loss = target_verification_loss(scores, self.threshold, label)
        return loss


def resolve_loss(targeted=False, task='CSI', threshold=None, loss_name='Entropy', confidence=0., clip_max=True):
    if task == 'SV' and loss_name == 'Entropy':
        loss = SpeakerVerificationLoss(targeted=targeted, threshold=threshold)
        grad_sign = -1
    elif task == 'SV' and loss_name == 'Margin':
        loss = SEC4SR_MarginLoss(targeted=targeted, confidence=confidence, task=task, threshold=threshold,
                                 clip_max=clip_max)
        grad_sign = -1
    else:
        loss = SEC4SR_CrossEntropy(reduction='none', task='CSI')  # ONLY FOR CSI TASK
        grad_sign = (1 - 2 * int(targeted))  # targeted: -1, untargeted: 1

    return loss, grad_sign


# def verification_loss(scores_EOT, y_batch_repeat):
#     # 0.6 threshold -1是为了适配cos
#     # y_batch_repeat 需要配合PGD接口
#     # 阈值跟模型相关
#     loss = -1 * (scores_EOT - 0.6)
#     return loss


def target_verification_loss(scores_EOT, threshold, y_batch_repeat):
    loss = threshold - scores_EOT
    return loss


def untarget_verification_loss(scores_EOT, threshold, y_batch_repeat):
    # score: 网络输出的余弦距离
    loss = scores_EOT - threshold
    return loss


def resolve_prediction(decisions):
    # print(decisions)
    predict = []
    for d in decisions:
        counts = Counter(d)
        predict.append(counts.most_common(1)[0][0])
    predict = np.array(predict)
    return predict
