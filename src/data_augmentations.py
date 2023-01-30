from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from timm.data.transforms_factory import create_transform

# TODO include mixup and rand as in
#  https://github.com/rwightman/pytorch-image-models/blob/29fda20e6d428bf636090ab207bbcf60617570ca/train.py
# rand_augment = rand_augment_transform(config_str='rand-m9-n3-mstd0.5')
# mixup = Mixup(mixup_alpha=1, cutmix_alpha=1, prob=1, switch_prob=1)

# TODO varying input size for our images, can't define function!!!
transform_test = create_transform(input_size=(), is_training=True, scale=(224, 224), hflip=0.5, vflip=0.5, re_prob=0.9)


resize_to_224x224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

randomAugment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=3, magnitude=9, num_magnitude_bins=31),
    transforms.ToTensor()
])

transformations = [transforms.Resize((224, 224)),
                   transforms.RandomHorizontalFlip(p=0.5),
                   transforms.RandomVerticalFlip(p=0.5),
                   transforms.RandomRotation(30),
                   transforms.ToTensor(),
                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
