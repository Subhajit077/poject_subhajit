
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from _model import TrafficSignCNN
from _config import batch_size, epochs, learning_rate, save_model_path
import os

def rename_files_with_valid_extensions(directory, valid_extension=".png"):
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file has an embedded extension like "_png"
            if "_png" in file or "_jpg" in file or "_jpeg" in file or "_bmp" in file:
                # Extract the base name and replace the embedded extension with a valid one
                new_file = file.replace("_png", ".png").replace("_jpg", ".jpg").replace("_jpeg", ".jpeg").replace("_bmp", ".bmp")
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, new_file)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

def load_data(train_dir):
    # Define transformations for the training dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32
        transforms.ToTensor(),       # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])
    
    # Load dataset from directory
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

def train_model(model, train_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    
    # Save the trained model
    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")
    print("Training complete!")

if __name__ == "__main__":
    # Define path to training dataset
    train_dir = "./_data/train"
    
    # Rename files with valid extensions
    rename_files_with_valid_extensions(train_dir)
    
    # Print directory contents
    print("Train directory contents:", os.listdir(train_dir))
    
    # Load data
    train_loader = load_data(train_dir)
    
    # Initialize model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficSignCNN().to(device)
    
    # Train the model
    train_model(model, train_loader, device)
