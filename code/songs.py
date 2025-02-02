#!/usr/bin/env python

import os
import torch
import torchaudio
import matplotlib.pyplot as plt

def compute_mel_spectrogram(waveform, sample_rate, n_fft=1024, hop_length=512, n_mels=128):
    """
    Compute a Mel-spectrogram from an audio waveform.
    
    Args:
        waveform (Tensor): Audio waveform tensor.
        sample_rate (int): Sample rate of the audio.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for STFT.
        n_mels (int): Number of Mel bands.
        
    Returns:
        Tensor: Log-scaled Mel-spectrogram.
    """
    # Create a MelSpectrogram transformer with a specified number of Mel bands.
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)
    # Apply a logarithmic transformation for better visualization.
    log_mel_spec = torch.log1p(mel_spec)
    return log_mel_spec

def load_and_process_song(file_path):
    """
    Loads an audio file and computes its log Mel-spectrogram.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        Tuple (Tensor, int): Log-scaled Mel-spectrogram and sample rate.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Song file not found: {file_path}")
    
    waveform, sample_rate = torchaudio.load(file_path)
    # Convert to mono if the audio is stereo.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    log_mel_spec = compute_mel_spectrogram(waveform, sample_rate)
    return log_mel_spec, sample_rate

def main():
    # Define the folder and expected filenames for the sample songs.
    songs_dir = 'data/songs'
    song_files = {
        'Rock': os.path.join(songs_dir, 'rock_sample.wav'),
        'Classical': os.path.join(songs_dir, 'classical_sample.wav'),
        'Pop': os.path.join(songs_dir, 'pop_sample.wav'),
        'Jazz': os.path.join(songs_dir, 'jazz_sample.wav')
    }
    
    # Create a 2x2 subplot for displaying the spectrograms.
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (genre, file_path) in enumerate(song_files.items()):
        try:
            log_mel_spec, sample_rate = load_and_process_song(file_path)
            # Ensure the spectrogram tensor is moved to CPU and converted to a NumPy array.
            spec_np = log_mel_spec.squeeze().cpu().numpy()
            im = axes[idx].imshow(spec_np, origin='lower', aspect='auto', cmap='viridis')
            axes[idx].set_title(f"{genre} (Sample Rate: {sample_rate} Hz)")
            axes[idx].set_xlabel("Time Frames")
            axes[idx].set_ylabel("Mel Frequency Bins")
            fig.colorbar(im, ax=axes[idx])
        except Exception as e:
            print(f"Error processing {genre} song: {e}")
            axes[idx].set_title(f"{genre} - Error")
    
    plt.tight_layout()
    
    # Ensure the results folder exists.
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'task_b_comparative_spectrograms.png')
    plt.savefig(output_path)
    print(f"Comparative spectrogram image saved to: {output_path}")
    plt.show()

if __name__ == '__main__':
    main()
