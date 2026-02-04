import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
from tqdm import tqdm

from models.vision_transformer import vit_small

class LinearClassifier(nn.Module):
    """Linear classifier for evaluation"""
    def __init__(self, dim, num_classes=1000):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)

@torch.no_grad()
def extract_features(model, data_loader):
    """Extract features from the model"""
    model.eval()
    features = []
    labels = []
    
    for images, targets in tqdm(data_loader, desc='Extracting features'):
        images = images.cuda(non_blocking=True)
        
        # Forward pass through backbone
        feats = model(images)
        
        features.append(feats.cpu())
        labels.append(targets)
    
    features = torch.cat(features)
    labels = torch.cat(labels)
    
    return features, labels

def evaluate_linear_probe(args):
    """Evaluate using linear probing"""
    
    # Build model
    print("Loading model...")
    model = vit_small(patch_size=args.patch_size)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Extract student backbone weights
    backbone_state = {k.replace('module.student.backbone.', ''): v 
                     for k, v in state_dict.items() 
                     if 'student.backbone' in k}
    
    model.load_state_dict(backbone_state, strict=False)
    model = model.cuda()
    model.eval()
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
    
    # Build datasets
    print("Building datasets...")
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_dataset = datasets.ImageFolder(args.train_data_path, transform=transform_train)
    val_dataset = datasets.ImageFolder(args.val_data_path, transform=transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Extract features
    print("Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader)
    
    print("Extracting validation features...")
    val_features, val_labels = extract_features(model, val_loader)
    
    # Build linear classifier
    print("Training linear classifier...")
    classifier = LinearClassifier(dim=model.embed_dim, num_classes=args.num_classes).cuda()
    
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Create data loaders for features
    train_dataset_feats = torch.utils.data.TensorDataset(train_features, train_labels)
    val_dataset_feats = torch.utils.data.TensorDataset(val_features, val_labels)
    
    train_loader_feats = DataLoader(train_dataset_feats, batch_size=1024, shuffle=True)
    val_loader_feats = DataLoader(val_dataset_feats, batch_size=1024, shuffle=False)
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        classifier.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for feats, targets in train_loader_feats:
            feats, targets = feats.cuda(), targets.cuda()
            
            outputs = classifier(feats)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        classifier.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for feats, targets in val_loader_feats:
                feats, targets = feats.cuda(), targets.cuda()
                outputs = classifier(feats)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        best_acc = max(best_acc, val_acc)
        
        print(f'Epoch {epoch}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%')
    
    print(f'\nFinal Best Accuracy: {best_acc:.2f}%')
    return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear probing evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    
    args = parser.parse_args()
    evaluate_linear_probe(args)