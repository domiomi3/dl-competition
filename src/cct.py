from torch import nn
from torch import Tensor
from vit_pytorch.cct import cct_7


class ModifiedCCTBase(nn.Module):
    def __init__(self):
        super().__init__()
        # Load CCT model pretrained on ImageNet-1K dataset
        self.cct = cct_7(arch='cct_7_7x2_224_sine', pretrained=True, progress=False)

    def disable_gradients(self, model) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x) -> Tensor:
        return self.cct(x)


class ModifiedCCTLinear(ModifiedCCTBase):
    def __init__(self, num_classes=10):
        super().__init__()
        self.disable_gradients(self.cct)
        # Load the pretrained model pretrained on Flowers102 data
        self.cct.classifier.fc = nn.Linear(in_features=256, out_features=num_classes)


class CCTBase(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.cct_7 = cct_7(arch='cct_7_3x1_32', pretrained=True, progress=False)

    def forward(self, x) -> Tensor:
        return self.cct(x)


class CCTLinear(ModifiedCCTBase):
    def __init__(self, num_classes=10):
        super().__init__()
        self.disable_gradients(self.cct)
        # Load the pretrained model pretrained on Flowers102 data
        self.cct.classifier.fc = nn.Linear(in_features=256, out_features=num_classes)


if __name__ == "__main__":
    model = ModifiedCCTLinear()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
