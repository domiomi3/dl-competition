from torchvision import transforms

resize_to_224x224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])

rand_augment = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(num_ops=3, magnitude=9, num_magnitude_bins=31),
    transforms.ToTensor()
])

basic_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
