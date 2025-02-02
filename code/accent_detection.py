#!/usr/bin/env python
"""
Accent Detection using a CNN in PyTorch

This script implements a simple accent detection system.
It includes:
- A custom Dataset class to load audio files and labels.
- A CNN model that processes Mel-spectrograms.
- A training loop to train the model.

Before running, ensure your accent detection data is arranged as follows:
- Place your audio files and a CSV file named 'labels.csv' in the folder: data/accent_dataset/
- The CSV should be formatted as:
    filename,label
    sample1.wav,0
    sample2.wav,1
    ...
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

# Custom dataset for accent detection
class AccentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory containing audio files and a labels.csv file.
            transform (callable, optional): Optional transformation to apply to the spectrogram.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        label_file = os.path.join(data_dir, 'labels.csv')
        
        # Expected CSV format: filename,label
        with open(label_file, 'r') as file:
            next(file)  # Skip header
            for line in file:
                filename, label = line.strip().split(',')
                self.samples.append((filename, int(label)))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        filename, label = self.samples[index]
        file_path = os.path.join(self.data_dir, filename)
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert waveform into a Mel-spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
        if self.transform:
            mel_spec = self.transform(mel_spec)
        
        # Return the spectrogram and label
        return mel_spec, label

# CNN model definition for accent detection
class AccentCNN(nn.Module):
    def __init__(self, num_classes):
        super(AccentCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Adjust the fully connected layer's input dimensions as per your spectrogram size.
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: [batch_size, 1, frequency, time]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs=20, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    print("Model training finished.")

if __name__ == '__main__':
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set the path to your accent detection dataset
    data_directory = 'data/accent_dataset'
    
    # Create the dataset and dataloader
    dataset = AccentDataset(data_directory)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Define the number of accent classes (adjust as necessary)
    num_accent_classes = 4
    model = AccentCNN(num_accent_classes).to(device)
    
    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=20, device=device)
    
    # Save the trained model to the code folder
    torch.save(model.state_dict(), os.path.join('code', 'accent_cnn_model.pth'))
