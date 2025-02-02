
## Important Note on the UrbanSound8K Dataset

Due to its large size, the UrbanSound8K dataset is **not included** in this repository. You must download and set up the dataset manually. Follow these steps:

1. **Download UrbanSound8K:**
   - Download the dataset from the official source (e.g., [UrbanSound8K Dataset](https://urbansounddataset.weebly.com/urbansound8k.html) or the link provided in your assignment instructions).
   
2. **Extract and Organize:**
   - Extract the downloaded archive.
   - Place the extracted folder in the `data/` directory and rename it to `UrbanSound8K` if necessary.
   - Ensure that the folder structure looks like this:
     ```
     data/UrbanSound8K/
     ├── metadata/
     │   └── UrbanSound8K.csv
     └── audio/
         ├── fold1/
         ├── fold2/
         └── ... (other fold folders)
     ```

3. **Optional Download Script:**
   - (Optional) You can create a download script (e.g., `download_urbansound8k.sh`) with the following content to automate the process:
     ```bash
     #!/bin/bash
     echo "Downloading UrbanSound8K dataset..."
     wget -O UrbanSound8K.tar.gz "YOUR_DATASET_DOWNLOAD_LINK_HERE"
     echo "Extracting dataset..."
     mkdir -p data/UrbanSound8K
     tar -xzf UrbanSound8K.tar.gz -C data/
     echo "UrbanSound8K dataset setup completed."
     ```
   - Replace `"YOUR_DATASET_DOWNLOAD_LINK_HERE"` with the actual download URL provided by your instructor or the dataset website.

## Environment Setup

1. **Python and Libraries:**
   - Install Python 3.x.
   - Install the required libraries using pip:
     ```
     pip install torch torchaudio numpy matplotlib pandas
     ```

2. **Data Preparation:**
   - **Accent Dataset:** Place your accent detection audio files and a `labels.csv` file inside `data/accent_dataset/`.  
     Make sure the filenames in `labels.csv` match your audio files (remember to change the names as needed).
   - **Sample Songs:** Place at least one sample song per genre (e.g., `rock_sample.wav`, `classical_sample.wav`, `pop_sample.wav`, `jazz_sample.wav`) inside `data/songs/`.

## Running the Code

- To train the accent detection model, run:
  ```bash
  python code/accent_detection.py
