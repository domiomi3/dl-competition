import timm
import torch


class EfficientNetv2SBase(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass through all layers of the model
        Args:
            x: input data
        Returns:
            Logits for each class
        """
        return self.model(x)

    def disable_gradients(self) -> None:
        """
        Freezes the layers of a model
        Args:
            model: The model with the layers to freeze
        Returns:
            None
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def get_num_params(self) -> int:
        """
        Counts learnable parameters
        Args:
            model: The model with parameters to count
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024 ** 2


class EfficientNetv2SFC(EfficientNetv2SBase):
    def __init__(self, num_classes=10):
        super().__init__()
        self.disable_gradients()
        self.model.classifier = torch.nn.Linear(in_features=1280, out_features=num_classes)


if __name__ == "__main__":
    model = EfficientNetv2SBase(10)
    print(model.get_size())
