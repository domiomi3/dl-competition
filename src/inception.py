import timm
import torch


class Inceptionv4Base(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.inc = timm.create_model('inception_v4', pretrained=True)
        self.disable_gradients(self.inc)
        self.inc.last_linear = torch.nn.Linear(in_features=1536, out_features=10)

    def forward(self, x) -> torch.Tensor:
        return self.inc(x)

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


if __name__ == "__main__":
    model = Inceptionv4Base()
    print(model, sum(p.numel() for p in model.parameters() if p.requires_grad))

