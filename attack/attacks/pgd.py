import torch
import torch.nn as nn

from ..attack import Attack
from ..models.loss import SpeakerVerificationLoss
import pdb
import torchaudio
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - x2: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - y: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of y`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_x2 = attack(x2, y)

    """

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=False):
        super().__init__("PGD", model)
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']
        self.loss = SpeakerVerificationLoss(threshold=self.model.threshold)

    def forward(self, x1, x2, y, runno, df = None):
        # df: a dataframe(dictionary) to store the results
        r"""
        Overridden.
        """

        x1 = x1.clone().detach().to(self.device)
        x2 = x2.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        # if self.targeted:
        #     target_y = self.get_target_label(x2, y)

        adv_x2 = x2.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_x2 = adv_x2 + \
                torch.empty_like(adv_x2).uniform_(-self.eps, self.eps)
            adv_x2 = torch.clamp(adv_x2, min=0, max=1).detach()

        # select best
        best_cost = -torch.inf
        best_decision = torch.tensor(0).float()
        best_adv_x2 = adv_x2.clone().detach()

        remaining_steps = self.steps
        attack_success = False
        sum_steps = 0
        while remaining_steps >= 0 and sum_steps < 400:
            if adv_x2.dim() != 3:
                assert adv_x2.dim() == 2
                adv_x2 = adv_x2.unsqueeze(0)
            adv_x2.requires_grad = True
            # decision, score = self.get_logits(adv_x2)
            try:
                decision, score = self.model(x1, adv_x2)
            except:
                decision, score = self.model.make_decision_SV(x1, adv_x2)
            # print(i, score)
            # Calculate loss
            cost = self.loss(score, y)
            # print(f'loss: {cost.item()}')
            # writer.add_scalar(f'Sample {runno} Loss', cost.item(), i)

            if attack_success:
                remaining_steps -= 1

            sum_steps += 1
            # Update adversarial x2
            grad = torch.autograd.grad(cost, adv_x2,
                                       retain_graph=False, create_graph=False)[0]

            adv_x2 = adv_x2.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x2 - x2,
                                min=-self.eps, max=self.eps)
            adv_x2 = torch.clamp(x2 + delta, min=-1, max=1).detach()

            # update best
            # if cost > best_cost:
            #     best_cost = cost
            #     best_adv_x2 = adv_x2.clone().detach()
            #     best_decision = decision.clone().detach()

            # detect attack successful
            if cost.item() > 0:
                # check if attack is successful
                torchaudio.save(f'a.wav', adv_x2.cpu().detach().squeeze(0), 16000, encoding='PCM_S', bits_per_sample=16)
                a = torchaudio.load(f'a.wav')[0].to(self.device)
                try:
                    decision, score = self.model(x1, a)
                except:
                    decision, score = self.model.make_decision_SV(x1, a)
                if decision.item() == 1:
                    attack_success = True
                    remaining_steps -= 1
            

        df['steps'].append(sum_steps)
        print(f'sum_steps: {sum_steps}')
        df['max_perturbation'].append(torch.max(torch.abs(adv_x2 - x2)).item())
        df['success'].append(attack_success)
        return adv_x2.clone().detach(), decision.clone().detach(), cost
        # return best_adv_x2, best_decision, best_cost
