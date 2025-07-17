import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_dataset(train=True, transform=None, data_dir='/storage/data/zhanghf/cifar10'):
    """
    Returns CIFAR-10 dataset.
    
    Args:
        train (bool): If True, returns training set, else test set
        transform: Optional transform to be applied on samples
        data_dir (str): Directory to store the dataset
    Returns:
        CIFAR-10 dataset
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=train, 
        download=False, 
        transform=transform
    )
    
    return dataset