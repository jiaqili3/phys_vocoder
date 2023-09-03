import torch

from ..attack import Attack
from .pgd import PGD
from .pgd_fixed import PGDFixed
import pdb
import torchaudio

class MultiAttack():
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
        adv_x2 = x2.clone().detach()

        cnt = 0
        attackers = []
        for m in self.models:
            pgd = PGDFixed(m, **self.kwargs)
            pgd.adver_dir = self.adver_dir
            attackers.append(pgd)
        remaining_steps = 0
        sum_steps = 0
        while remaining_steps >= 0 and sum_steps < 400:
            is_all_success = [True]*2

            pgd1 = attackers[0]
            pgd2 = attackers[1]
            adv_x2_1, _, cost, sum_steps1 = pgd1(x1, adv_x2.clone().detach(), y)
            adv_x2_2, _, _, sum_steps2 = pgd2(x1, adv_x2.clone().detach(), y)
            adv_x2 = (adv_x2_1 + adv_x2_2 ) / 2
            delta = torch.clamp(adv_x2 - x2, min=-pgd1.eps, max=pgd1.eps)
            adv_x2 = torch.clamp(x2 + delta, min=-1, max=1).detach()
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
            # print(cnt, decision1.item(), decision2.item())
            if decision1.item() == 0:
                is_all_success[0] = False
            else:
                is_all_success[0] = True
            if decision2.item() == 0:
                is_all_success[1] = False
            else:
                is_all_success[1] = True
            cnt += 1
            # if df is not None:
            #     df['steps'].append(cnt)
            #     print(f'sum_steps: {cnt}')
            #     df['max_perturbation'].append(torch.max(torch.abs(adv_x2 - x2)).item())
            #     print(f'max_perturbation: {torch.max(torch.abs(adv_x2 - x2)).item()}')
            #     df['success'].append(is_all_success)
            # return adv_x2, torch.ones(1) if is_all_success[0] and is_all_success[1] else torch.zeros(1), cnt, cnt
            if is_all_success == [True]*2:
                remaining_steps -= 1
        if df is not None:
            df['steps'].append(cnt)
            print(f'sum_steps: {cnt}')
            df['max_perturbation'].append(torch.max(torch.abs(adv_x2 - x2)).item())
            print(f'max_perturbation: {torch.max(torch.abs(adv_x2 - x2)).item()}')
            df['success'].append(is_all_success)
        # pdb.set_trace()
        print('complete')
        return adv_x2, torch.ones(1), cnt, cnt
            # check success
            # check if 

        while True:
            for _, attack in enumerate(self.attacks):
                adv_eval_waveforms[fails_index], _, _, _ = attack(enroll_waveforms[fails_index], adv_eval_waveforms[fails_index], labels[fails_index])

            # check succeeds
            is_successes = torch.zeros(batch_size, len(self.attacks)).to(self.device)
            for i, attack in enumerate(self.attacks):
                enroll_embeddings = attack.model(enroll_waveforms[fails_index])
                adv_eval_embeddings = attack.model(adv_eval_waveforms[fails_index])
                similarity_scores[:, i] = attack.score_function(enroll_embeddings, adv_eval_embeddings)
                decisions = attack.decision_function(enroll_embeddings, adv_eval_embeddings)
                is_successes[:, i] = torch.logical_xor(decisions, labels[fails_index])
            is_successes = torch.min(is_successes, dim=1)[0].bool()  # if all attacks succeed, then the adversarial waveform is successful
            fails_mask = ~is_successes
            fails_index = torch.masked_select(fails_index, fails_mask)

            if self.until_all_success:
                if len(fails_index) == 0:
                    break
            else:
                break

        return adv_eval_waveforms, is_successes, similarity_scores
