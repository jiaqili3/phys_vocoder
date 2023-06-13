import torch

class PGDAttack(torch.nn.Module):
    def __init__(self, surrogate_models: list, *args, **kwargs) -> None:
        super().__init__()
        self.surrogate_models = surrogate_models
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

