import sys
sys.path.append('.')

from .AttackTypes import AttackTypes
from .utils import resolve_loss, resolve_prediction
import numpy as np
import torch
from .adaptive_attack.EOT import EOT


class FGSM():

    def __init__(self, model, epsilon=0.002, loss='Entropy', targeted=False,
                 batch_size=1, EOT_size=1, EOT_batch_size=1,
                 verbose=1):

        self.model = model  # remember to call model.eval()
        self.epsilon = epsilon
        self.loss_name = loss
        self.targeted = targeted
        self.batch_size = batch_size
        EOT_size = max(1, EOT_size)
        EOT_batch_size = max(1, EOT_batch_size)
        assert EOT_size % EOT_batch_size == 0, 'EOT size should be divisible by EOT batch size'
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose

        self.threshold = None

        self.loss, self.grad_sign = resolve_loss(targeted=self.targeted)
        self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, True)

        self.max_iter = 1  # FGSM is single step attack, keep in consistency with PGD
        self.step_size = epsilon  # keep in consistency with PGD

    def attack_batch_unified(self, xs: list, y_batch, lower, upper, batch_id, logger,
                             attack_type=AttackTypes.VERIFICATION):
        # unify different attack types: identification and verification and ensemble task
        num_x = len(xs)
        if attack_type == AttackTypes.IDENTIFICATION:
            assert num_x == 1, 'verification task only support one input'
        elif attack_type == AttackTypes.VERIFICATION:
            assert num_x == 2, 'identification task receives 2 inputs'
        elif attack_type == AttackTypes.VERIFICATION_ENSEMBLE:
            assert num_x == 2, 'identification ensemble task receives 2 inputs'

        x1_batch = xs[0].clone()  # avoid influcing
        x1_batch.requires_grad = True
        if num_x == 2:
            x2_batch = xs[1].clone()
            x2_batch.requires_grad = True
            x1_batch.requires_grad = False
        # x_batch.retain_grad()
        success = None
        if type(self.model) != list:
            self.model = [self.model]

        for iter in range(self.max_iter + 1):
            EOT_num_batches = int(self.EOT_size // self.EOT_batch_size) if iter < self.max_iter else 1
            real_EOT_batch_size = self.EOT_batch_size if iter < self.max_iter else 1
            use_grad = True if iter < self.max_iter else False

            # record models data for ensemble
            predicts = []
            targets = []
            losses = []

            for model in self.model:

                if num_x == 1:
                    scores, loss, grad, decisions = self.EOT_wrapper(x1_batch, y_batch, EOT_num_batches,
                                                                     real_EOT_batch_size, use_grad)
                elif num_x == 2:
                    scores, loss, grad, decisions = self.EOT_wrapper(x1_batch, x2_batch, y_batch, EOT_num_batches,
                                                                     real_EOT_batch_size, use_grad, model)
                scores.data = scores / EOT_num_batches
                loss.data = loss / EOT_num_batches
                if iter < self.max_iter:
                    grad.data = grad / EOT_num_batches
                # predict = torch.argmax(scores.data, dim=1).detach().cpu().numpy()
                predict = resolve_prediction(decisions)
                target = y_batch.detach().cpu().numpy()
                success = self.compare(target, predict, self.targeted)
                predicts.append(predict)
                targets.append(target)
                losses.append(loss.detach().cpu().numpy().tolist())

                if self.verbose:
                    if attack_type == AttackTypes.VERIFICATION_ENSEMBLE:
                        info = 'iter:{} losses:{} predicts: {}, targets: {}'.format(iter, losses, predicts, targets)
                    else:
                        info = "batch:{} iter:{} loss: {} predict: {}, target: {}".format(batch_id, iter,
                                                                                          loss.detach().cpu().numpy().tolist(),
                                                                                          predict, target)
                    if logger is not None:
                        logger.info(info)
                    else:
                        print(info)

                if iter < self.max_iter:
                    if attack_type == AttackTypes.IDENTIFICATION:
                        x1_batch.grad = grad
                        # x_batch.data += self.epsilon * torch.sign(x_batch.grad) * self.grad_sign
                        # x_batch.data += self.epsilon * torch.sign(grad) * self.grad_sign
                        x1_batch.data += self.step_size * torch.sign(x1_batch.grad) * self.grad_sign
                        x1_batch.grad.zero_()
                        # x_batch.data = torch.clamp(x_batch.data, min=lower, max=upper)
                        x1_batch.data = torch.min(torch.max(x1_batch.data, lower), upper)

                    elif attack_type == AttackTypes.VERIFICATION or attack_type == AttackTypes.VERIFICATION_ENSEMBLE:
                        x2_batch.grad = grad
                        # x_batch.data += self.epsilon * torch.sign(x_batch.grad) * self.grad_sign
                        # x_batch.data += self.epsilon * torch.sign(grad) * self.grad_sign
                        x2_batch.data += self.step_size * torch.sign(x2_batch.grad) * self.grad_sign
                        x2_batch.grad.zero_()
                        # x_batch.data = torch.clamp(x_batch.data, min=lower, max=upper)
                        x2_batch.data = torch.min(torch.max(x2_batch.data, lower), upper)

        if attack_type == AttackTypes.VERIFICATION_ENSEMBLE:
            success = []
            for model in self.model:
                decision, _ = model.make_decision_SV(x1_batch, x2_batch)
                decision = decision.detach().cpu().item()
                if decision == 1:
                    success.append(True)
                else:
                    success.append(False)
            return x2_batch, [all(success)]

        if num_x == 2:
            return x2_batch, success
        elif num_x == 1:
            return x1_batch, success

    def attack_batch(self, x_batch, y_batch, lower, upper, batch_id):
        return self.attack_batch_unified([x_batch], y_batch, lower, upper, batch_id, None, AttackTypes.IDENTIFICATION)



    def attack_sv_batch(self, x1_batch, x2_batch, y_batch, lower, upper, batch_id, logger):
        return self.attack_batch_unified([x1_batch, x2_batch], y_batch, lower, upper, batch_id, logger,
                                         AttackTypes.VERIFICATION)


    def attack_sv_ensemble_batch(self, x1_batch, x2_batch, y_batch, lower, upper, batch_id, logger):
        return self.attack_batch_unified([x1_batch, x2_batch], y_batch, lower, upper, batch_id, logger,
                                         AttackTypes.VERIFICATION_ENSEMBLE)

    def attack(self, x, y):

        lower = -1
        upper = 1
        assert lower <= x.max() < upper, 'generating adversarial examples should be done in [-1, 1) float domain'
        n_audios, n_channels, _ = x.size()
        assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal'
        lower = torch.tensor(lower, device=x.device, dtype=x.dtype).expand_as(x)
        upper = torch.tensor(upper, device=x.device, dtype=x.dtype).expand_as(x)

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))
        for batch_id in range(n_batches):
            x_batch = x[batch_id * batch_size:(batch_id + 1) * batch_size]  # (batch_size, 1, max_len)
            y_batch = y[batch_id * batch_size:(batch_id + 1) * batch_size]
            lower_batch = lower[batch_id * batch_size:(batch_id + 1) * batch_size]
            upper_batch = upper[batch_id * batch_size:(batch_id + 1) * batch_size]
            adver_x_batch, success_batch = self.attack_batch(x_batch, y_batch, lower_batch, upper_batch, batch_id)
            if batch_id == 0:
                adver_x = adver_x_batch
                success = success_batch
            else:
                adver_x = torch.cat((adver_x, adver_x_batch), 0)
                success += success_batch

        return adver_x, success
