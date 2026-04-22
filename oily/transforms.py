from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_inference_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])