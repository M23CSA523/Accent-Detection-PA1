# Accent-Detection-PA1

## Project Overview

This repository contains the code, documentation, and results for the Speech Understanding Programming Assignment – 1. The project is divided into two main parts:

1. **Group Assignment (Question 1): Accent Detection**  
   - Analysis of accent detection including its importance, current approaches, evaluation metrics, and open challenges.
   - A PyTorch implementation of a CNN-based accent detection system.

2. **Individual Assignment (Question 2): Spectrograms and Windowing Techniques**  
   - Experiments with the UrbanSound8K dataset.
   - Implementation of different windowing functions (Hann, Hamming, and Rectangular) for STFT.
   - Comparative analysis of spectrograms for four songs from different genres.
   - A simple classifier using features extracted from the spectrograms.

## Repository Structure
Accent-Detection-PA1/
├── README.md            # Project overview and instructions.
├── report.pdf           # Detailed report covering both assignments.
├── presentation.pdf     # Slide deck for the project presentation.
├── code/
│   ├── accent_detection.py          # Code for the Accent Detection task.
│   ├── windowing_and_classification.py  # Code for spectrogram analysis and classification.
│   └── [Additional helper scripts]
├── data/
│   ├── accent_dataset/  # Accent detection dataset and associated CSV file.
│   ├── UrbanSound8K/    # UrbanSound8K dataset (or directions to download it).
│   └── songs/           # Sample songs from various genres.
└── results/             # Folder to store generated spectrogram images and outputs.


## Instructions

1. **Environment Setup:**
   - Install Python 3.x.
   - Install required libraries using pip:
     ```
     pip install torch torchaudio numpy matplotlib pandas
     ```

2. **Data Preparation:**
   - Place your accent detection audio files and a `labels.csv` file inside `data/accent_dataset/`.  
     The CSV should have two columns: `filename` and `label` (without quotes).
   - Download and place the UrbanSound8K dataset (or provide instructions) in `data/UrbanSound8K/`.
   - Add at least one sample song per genre (e.g., rock, classical, pop, jazz) inside `data/songs/`.

3. **Running the Code:**
   - To train the accent detection model, run:
     ```
     python code/accent_detection.py
     ```
   - To run spectrogram generation, windowing experiments, and classification on UrbanSound8K, run:
     ```
     python code/windowing_and_classification.py
     ```

4. **Results:**
   - All generated images (spectrograms, etc.) will be saved inside the `results/` folder.

5. **Submission:**
   - Zip the entire repository folder as `Rollno_PA1.zip` and submit as per the assignment instructions.

