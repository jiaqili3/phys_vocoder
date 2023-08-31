import torch

from ..attack import Attack
from .pgd import PGD
import pdb
import torchaudio

class JointAttack():
    r"""
    MultiAttack is a class to attack a model with various attacks agains same images and labels.

    Arguments:
        model (nn.Module): model to attack.
        attacks (list): list of attacks.

    Examples::
        >>> atk1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk = torchattacks.MultiAttack([atk1, atk2])
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, **kwargs):
        # super().__init__("PGD", model[0])
        self.models = model
        self.kwargs = kwargs

    def check_validity(self):
        if len(self.attacks) < 2:
            raise ValueError("More than two attacks should be given.")

        # ids = [id(attack.model) for attack in self.attacks]
        # print(ids)
        # if len(set(ids)) != 1:
        #     raise ValueError("At least one of attacks is referencing a different model.")

    def forward(self, x1, x2, y, runno=None, df=None):
        r"""
        Overridden.
        """
        x1 = x1.clone().detach().to(self.device)
        x2 = x2.clone().detach().to(self.device)
        y = y.clone().detach().to(self.device)

        pgd1 = PGD(self.models[0], **self.kwargs)
        pgd1.adver_dir = self.adver_dir
        pgd2 = PGD(self.models[1], **self.kwargs)
        pgd2.adver_dir = self.adver_dir

        adv_x2 = x2.clone().detach()
        remaining_steps = self.kwargs['steps']
        self.eps = self.kwargs['eps']
        attack_success = False
        sum_steps = 0

        while remaining_steps >= 0 and sum_steps < 400:
            adv_x2_1, _, cost, sum_steps1 = pgd1(x1, adv_x2.clone().detach(), y)
            delta_1 = adv_x2_1 - adv_x2
            
            adv_x2_2, _, _, sum_steps2 = pgd2(x1, adv_x2.clone().detach(), y)
            delta_2 = adv_x2_2 - adv_x2
            
            # select the intersection of two deltas (where the two values are equal)
            delta = torch.where(delta_1 == delta_2, delta_1, 0)
            delta = torch.clamp(adv_x2+delta-x2, min=-self.eps, max=self.eps)
            adv_x2 = torch.clamp(x2+delta, min=-1, max=1).detach()
            sum_steps += 1
            # print(sum_steps)
            # print(torch.max(delta))

            # check if attack is successful
            torchaudio.save(f'{self.adver_dir}/a.wav', adv_x2.cpu().detach().squeeze(0), 16000, encoding='PCM_S', bits_per_sample=16)
            a = torchaudio.load(f'{self.adver_dir}/a.wav')[0].to(self.device).unsqueeze(0)
            try:
                decision1, score = self.models[0](x1, a)
            except:
                decision1, score = self.models[0].make_decision_SV(x1, a)
            try:
                decision2, score = self.models[1](x1, a)
            except:
                decision2, score = self.models[1].make_decision_SV(x1, a)
            if decision1.item() == 1 and decision2.item() == 1:
                attack_success = True
                remaining_steps -= 1

        if df is not None:
            df['steps'].append(sum_steps)
            print(f'sum_steps: {sum_steps}')
            df['max_perturbation'].append(torch.max(torch.abs(adv_x2 - x2)).item())
            df['success'].append(attack_success)

        return adv_x2.clone().detach(), decision1.clone().detach(), cost, sum_steps

