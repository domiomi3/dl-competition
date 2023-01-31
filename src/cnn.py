""" File with CNN models. Add your custom CNN model here. """
import timm

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class SampleModel(nn.Module):
    """
    A sample PyTorch CNN model
    """
    def __init__(self, input_shape=(3, 224, 224), num_classes=10):
        super(SampleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=10, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(in_features=60500, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x


class EfficientNetv2SBase(nn.Module):
    def __init__(self, num_classes=10, dropout=0):
        super().__init__()
        self.model = timm.create_model("tf_efficientnetv2_s", pretrained=True, num_classes=num_classes,
                                       drop_rate=dropout)

    def forward(self, x) -> Tensor:
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
        Returns:
            None
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def get_num_params(self) -> int:
        """
        Counts learnable parameters
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_size(self) -> int:
        """
        Computes model size
        Returns:
            Model size
        """
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024 ** 2


class EfficientNetv2STuned(EfficientNetv2SBase):
    def __init__(self, num_classes=10, dropout=0):
        super().__init__(num_classes, dropout)
        self.disable_gradients()
        self.model.classifier = nn.Linear(in_features=1280, out_features=num_classes)
