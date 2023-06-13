import sys
from attack.attacks.adaptive_attack.EOT import EOT, EOTSV, EOTSVEnsemble
from .FGSM import FGSM
from .AttackTypes import AttackTypes
from .utils import resolve_loss
import numpy as np
import torch


class PGD(FGSM):

    def __init__(self, model, task='CSI', epsilon=0.002, step_size=0.0004, max_iter=10, num_random_init=0,
                 targeted=False, batch_size=1, EOT_size=1, EOT_batch_size=1, verbose=1):

        self.model = model  # remember to call model.eval()
        self.task = task
        self.epsilon = epsilon
        self.step_size = step_size
        self.max_iter = max_iter
        self.num_random_init = num_random_init
        self.targeted = targeted
        self.batch_size = batch_size
        EOT_size = max(1, EOT_size)
        EOT_batch_size = max(1, EOT_batch_size)
        assert EOT_size % EOT_batch_size == 0, 'EOT size should be divisible by EOT batch size'
        self.EOT_size = EOT_size
        self.EOT_batch_size = EOT_batch_size
        self.verbose = verbose
        assert EOT_size == 1, 'EOT size should be 1'
        assert EOT_batch_size == 1, 'EOT batch size should be 1'

        self.threshold = None
        if self.task == 'SV':
            self.loss, self.grad_sign = resolve_loss(targeted=self.targeted, task=self.task,
                                                     threshold=self.model.threshold)
            self.threshold = self.model.threshold
            self.EOT_wrapper = EOTSV(self.model, self.loss, self.EOT_size, self.EOT_batch_size, use_grad=True)
        elif self.task == 'EnsembleSV':
            self.grad_sign = -1
            self.EOT_wrapper = EOTSVEnsemble(self.model, self.EOT_size, self.EOT_batch_size, True)
        else:
            self.EOT_wrapper = EOT(self.model, self.loss, self.EOT_size, self.EOT_batch_size, True)

    def attack_unified(self, xs, y, attack_type=AttackTypes.VERIFICATION):
        num_x = len(xs)
        if attack_type == AttackTypes.IDENTIFICATION:
            assert num_x == 1, 'identification task only support one input'
        elif attack_type == AttackTypes.VERIFICATION:
            assert num_x == 2, 'verification task receives 2 inputs'
        elif attack_type == AttackTypes.VERIFICATION_ENSEMBLE:
            assert num_x == 2, 'verification ensemble task receives 2 inputs'

        lower_setting = -1
        upper_setting = 1

        x1 = xs[0]
        x_ori = x1.clone()
        n_audios, n_channels, max_len = x1.size()
        upper = torch.clamp(x1 + self.epsilon, max=upper_setting)
        lower = torch.clamp(x1 - self.epsilon, min=lower_setting)
        if num_x == 2:
            x2 = xs[1]
            x_ori = x2.clone()
            n_audios, n_channels, max_len = x2.size()
            upper = torch.clamp(x2 + self.epsilon, max=upper_setting)
            lower = torch.clamp(x2 - self.epsilon, min=lower_setting)

        assert lower_setting <= x1.max() < upper_setting, 'generating adversarial examples should be done in [-1, 1) float domain'
        if num_x == 2:
            assert lower_setting <= x2.max() < upper_setting, 'generating adversarial examples should be done in [-1, 1) float domain'

        # n_audios, n_channels, max_len = x.size()
        # assert n_channels == 1, 'Only Support Mono Audio'
        assert y.shape[0] == n_audios, 'The number of x and y should be equal'

        batch_size = min(self.batch_size, n_audios)
        n_batches = int(np.ceil(n_audios / float(batch_size)))

        best_success_rate = -1
        best_success = None
        best_adver_x = None
        for init in range(max(1, self.num_random_init)):
            # if self.num_random_init > 0:
            #     # x = x_ori + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, \
            #     #                                            (n_audios, n_channels, max_len)), device=x.device,
            #     #                          dtype=x.dtype)
            #     x2 = x_ori + torch.tensor(np.random.uniform(-self.epsilon, self.epsilon, \
            #                                                (n_audios, n_channels, max_len)), device=x2.device,
            #                              dtype=x2.dtype)
            for batch_id in range(n_batches):
                x1_batch = x1[batch_id * batch_size:(batch_id + 1) * batch_size]  # (batch_size, 1, max_len)
                if num_x == 2:
                    x2_batch = x2[batch_id * batch_size:(batch_id + 1) * batch_size]  # (batch_size, 1, max_len)
                y_batch = y[batch_id * batch_size:(batch_id + 1) * batch_size]  # verification: [0 or 1]

                lower_batch = lower[batch_id * batch_size:(batch_id + 1) * batch_size]
                upper_batch = upper[batch_id * batch_size:(batch_id + 1) * batch_size]

                if attack_type == AttackTypes.VERIFICATION:
                    adver_x_batch, success_batch = self.attack_sv_batch(x1_batch, x2_batch, y_batch, lower_batch,
                                                                        upper_batch, '{}-{}'.format(init, batch_id),
                                                                        self.logger)
                elif attack_type == AttackTypes.IDENTIFICATION:
                    adver_x_batch, success_batch = self.attack_batch(x1_batch, y_batch, lower_batch, upper_batch,
                                                                     '{}-{}'.format(init, batch_id))
                elif attack_type == AttackTypes.VERIFICATION_ENSEMBLE:
                    adver_x_batch, success_batch = self.attack_sv_ensemble_batch(x1_batch, x2_batch, y_batch,
                                                                                 lower_batch,
                                                                                 upper_batch,
                                                                                 '{}-{}'.format(init, batch_id),
                                                                                 self.logger)
                if batch_id == 0:
                    adver_x = adver_x_batch
                    success = success_batch
                else:
                    adver_x = torch.cat((adver_x, adver_x_batch), 0)
                    success += success_batch
            if sum(success) / len(success) > best_success_rate:
                best_success_rate = sum(success) / len(success)
                best_success = success
                best_adver_x = adver_x

        return best_adver_x, best_success

    def attack(self, x, y):
        return self.attack_unified([x], y, AttackTypes.IDENTIFICATION)

    def attack_sv(self, x1, x2, y):
        return self.attack_unified([x1, x2], y, AttackTypes.VERIFICATION)

    def attack_sv_ensemble(self, x1, x2, y):
        return self.attack_unified([x1, x2], y, AttackTypes.VERIFICATION_ENSEMBLE)
