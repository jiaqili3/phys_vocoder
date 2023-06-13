import torch
import torch.nn as nn

from ..attack import Attack


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

    def __init__(self, model, eps=8/255, alpha=2/255, steps=10, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ['default', 'targeted']

    def forward(self, x1, x2, y):
        r"""
        Overridden.
        """

        x1 = x1.clone().detach().to(self.device)
        x2 = x2.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        if self.targeted:
            target_y = self.get_target_label(x2, y)

        loss = nn.CrossEntropyLoss()
        adv_x2 = x2.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_x2 = adv_x2 + \
                torch.empty_like(adv_x2).uniform_(-self.eps, self.eps)
            adv_x2 = torch.clamp(adv_x2, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_x2.requires_grad = True
            decision, score = self.get_logits(adv_x2)

            # Calculate loss
            if self.targeted:
                cost = -loss(score, target_y)
            else:
                cost = loss(score, y)

            # Update adversarial x2
            grad = torch.autograd.grad(cost, adv_x2,
                                       retain_graph=False, create_graph=False)[0]

            adv_x2 = adv_x2.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_x2 - x2,
                                min=-self.eps, max=self.eps)
            adv_x2 = torch.clamp(x2 + delta, min=0, max=1).detach()

        return adv_x2
