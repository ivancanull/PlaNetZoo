import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import time
from tqdm import tqdm
import argparse

from src.data.cifar10 import get_cifar10_dataset
from src.planetzoo.model.cnn.alexnet import AlexNet


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.3f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def main():
    # Argument parser for checkpoint resume
    parser = argparse.ArgumentParser(description='Train AlexNet on CIFAR-10 with checkpoint resume')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Training configuration
    config = {
        'batch_size': 128,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 5e-4,
        'num_workers': 4,
        'save_dir': './checkpoints',
        'save_every': 10,  # Save model every N epochs
    }
    # Add TensorBoard SummaryWriter for monitoring
    from torch.utils.tensorboard.writer import SummaryWriter

    writer = SummaryWriter(log_dir=os.path.join(config['save_dir'], 'tensorboard'))

    # Helper function to log gradients, weights, and loss
    def log_tensorboard(writer, model, loss, epoch, step, phase='train'):
        # Log loss
        writer.add_scalar(f'{phase}/loss', loss, step)
        # Log weights and gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'{phase}/weights/{name}', param.data.cpu().numpy(), step)
                if param.grad is not None:
                    writer.add_histogram(f'{phase}/grads/{name}', param.grad.cpu().numpy(), step)

    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms for CIFAR-10
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    train_dataset = get_cifar10_dataset(train=True, transform=train_transform)
    val_dataset = get_cifar10_dataset(train=False, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model for CIFAR-10 (10 classes, smaller input size)
    model = AlexNet(
        num_classes=10,
        input_channels=3,
        hidden_channels=[64, 192, 384, 256, 256],
        kernel_sizes=[3, 3, 3, 3, 3],  # Smaller kernels for 32x32 images
        hstrides=[1, 1, 1, 1, 1],      # Smaller strides
        wstrides=[1, 1, 1, 1, 1],
        linear_sizes=[1024, 512],      # Smaller linear layers
        dropout=0.5
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    best_val_acc = 0.0
    start_epoch = 0

    # Optionally resume from a checkpoint
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"=> Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
    elif args.resume is not None:
        print(f"=> No checkpoint found at '{args.resume}', starting from scratch.")

    print("Starting training...")
    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['save_every'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
