#!/usr/bin/env python
"""
Windowing and Spectrogram Classification for UrbanSound8K

This script performs the following:
1. Computes log-scaled spectrograms using different window functions (Hann, Hamming, Rectangular) via STFT.
2. Visualizes and saves the generated spectrogram images.
3. Loads the UrbanSound8K dataset from the proper directory and trains a simple CNN classifier on the spectrogram features.
4. Generates and saves spectrograms for sample songs from different genres.

IMPORTANT: 
- Ensure that the UrbanSound8K dataset is placed in the folder: data/UrbanSound8K/
- Ensure that sample song files (e.g., rock_sample.wav, classical_sample.wav, pop_sample.wav, jazz_sample.wav) are in data/songs/
- This script assumes that the UrbanSound8K metadata CSV file is located at: data/UrbanSound8K/metadata/UrbanSound8K.csv
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# -----------------------------
# Function to Compute Log-Scaled Spectrograms
# -----------------------------
def compute_log_spectrogram(waveform, sample_rate, n_fft=1024, hop_length=512, window_type='hann'):
    """
    Computes the log-scaled spectrogram using STFT with the specified window function.
    
    Args:
        waveform (Tensor): Audio waveform tensor.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for STFT.
        window_type (str): Type of window ('hann', 'hamming', or 'rectangular').
        
    Returns:
        Tensor: Log-scaled spectrogram.
    """
    if window_type.lower() == 'hann':
        window = torch.hann_window(n_fft)
    elif window_type.lower() == 'hamming':
        window = torch.hamming_window(n_fft)
    elif window_type.lower() == 'rectangular':
        window = torch.ones(n_fft)
    else:
        raise ValueError("Invalid window type. Choose 'hann', 'hamming', or 'rectangular'.")
    
    stft_result = torch.stft(waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    magnitude = torch.abs(stft_result)
    log_spectrogram = torch.log1p(magnitude)
    return log_spectrogram

# -----------------------------
# Utility Function to Display and Save Spectrograms
# -----------------------------
def display_spectrogram(spectrogram, title, save_path=None):
    """
    Displays and optionally saves a spectrogram image.
    
    Args:
        spectrogram (Tensor): Spectrogram tensor.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the image file.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram.squeeze().numpy(), origin='lower', aspect='auto', cmap='viridis')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(label="Log Magnitude")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# -----------------------------
# UrbanSound8K Dataset Class
# -----------------------------
class UrbanSoundDataset(Dataset):
    def __init__(self, metadata_csv, data_dir, window_type='hann', transform=None):
        """
        Args:
            metadata_csv (str): Path to the UrbanSound8K metadata CSV file.
            data_dir (str): Base directory containing audio files (organized by folds).
            window_type (str): Window function type for STFT ('hann', 'hamming', 'rectangular').
            transform (callable, optional): Optional transformation to apply to the spectrogram.
            
        IMPORTANT:
            - The UrbanSound8K dataset should be located in data/UrbanSound8K/
            - The metadata CSV file is expected at: data/UrbanSound8K/metadata/UrbanSound8K.csv
        """
        self.metadata = pd.read_csv(metadata_csv)
        self.data_dir = data_dir
        self.window_type = window_type
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        fold = f"fold{row['fold']}"
        file_path = os.path.join(self.data_dir, fold, row['slice_file_name'])
        waveform, sample_rate = torchaudio.load(file_path)
        # If stereo, convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        log_spec = compute_log_spectrogram(waveform, sample_rate, window_type=self.window_type)
        if self.transform:
            log_spec = self.transform(log_spec)
        label = row['classID']
        return log_spec, label

# -----------------------------
# CNN Model for UrbanSound8K Classification
# -----------------------------
class UrbanSoundCNN(nn.Module):
    def __init__(self, num_classes):
        super(UrbanSoundCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Adjust the fully connected layer's dimensions based on your spectrogram size.
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# Training Function for the Classifier
# -----------------------------
def train_classifier(model, dataloader, criterion, optimizer, num_epochs=15, device='cpu'):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for specs, labels in dataloader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * specs.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    print("Classifier training completed.")

# -----------------------------
# Main Routine for Windowing, Classification, and Genre Analysis
# -----------------------------
if __name__ == '__main__':
    # PART A: UrbanSound8K Windowing Experiments
    metadata_csv = 'data/UrbanSound8K/metadata/UrbanSound8K.csv'
    audio_dir = 'data/UrbanSound8K/audio'
    
    # Verify that the UrbanSound8K dataset is available.
    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}. Please ensure the UrbanSound8K dataset is correctly placed.")
    
    # Visualize spectrograms for a sample file using different window functions.
    sample_file = os.path.join(audio_dir, 'fold1', '101415-3-0-2.wav')
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file not found at {sample_file}. Check your UrbanSound8K data placement.")
    
    sample_waveform, sr = torchaudio.load(sample_file)
    sample_waveform = sample_waveform[:, :sr]  # Use a 1-second clip
    
    for win in ['hann', 'hamming', 'rectangular']:
        log_spec = compute_log_spectrogram(sample_waveform, sr, window_type=win)
        title = f"{win.capitalize()} Window Spectrogram"
        save_file = os.path.join('results', f'spectrogram_{win}.png')
        display_spectrogram(log_spec, title, save_path=save_file)
    
    # PART A Continued: Train a classifier using one window type (here, using 'hann')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected_window = 'hann'
    dataset = UrbanSoundDataset(metadata_csv, audio_dir, window_type=selected_window)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    num_classes = 10  # UrbanSound8K contains 10 classes
    model = UrbanSoundCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training classifier using features from the {selected_window} window...")
    train_classifier(model, dataloader, criterion, optimizer, num_epochs=15, device=device)
    torch.save(model.state_dict(), os.path.join('code', f'urbansound_cnn_{selected_window}.pth'))
    
    # PART B: Comparative Analysis for Songs from Different Genres
    # IMPORTANT: Ensure sample songs are placed in data/songs/ with the following filenames.
    song_files = {
        'rock': 'data/songs/rock_sample.wav',
        'classical': 'data/songs/classical_sample.wav',
        'pop': 'data/songs/pop_sample.wav',
        'jazz': 'data/songs/jazz_sample.wav'
    }
    
    for genre, filepath in song_files.items():
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found. Please add the {genre} sample song in data/songs/.")
            continue
        try:
            waveform, sr = torchaudio.load(filepath)
            log_spec = compute_log_spectrogram(waveform, sr, window_type='hann')
            title = f"{genre.capitalize()} Genre - Hann Window Spectrogram"
            save_file = os.path.join('results', f'{genre}_spectrogram.png')
            display_spectrogram(log_spec, title, save_path=save_file)
        except Exception as e:
            print(f"Error processing {genre} sample: {e}")
