from torchvision import transforms

resize_to_224x224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

resize_and_colour_jitter = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor()
])
