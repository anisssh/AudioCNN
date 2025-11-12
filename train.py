from pathlib import Path
import modal
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim

from model import AudioCNN

app = modal.App("audio-cnn")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip", "ffmpeg", "libsndfile1"])
         .run_commands(["cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O  esc50.zip",
                        "cd /tmp && unzip esc50.zip",
                        "mkdir -p /opt/esc50-data",
                        "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
                        "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.split = split
        self.transform = transform

        if split == "train":
            self.metadata = self.metadata[self.metadata['fold'] != 5]
        else:
            self.metadata = self.metadata[self.metadata['fold'] == 5]

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.metadata['label'] = self.metadata['category'].map(
            self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir/"audio"/row["filename"]

        # samples = sample rate * duration

        waveform, sample_rate = torchaudio.load(
            audio_path)  # returns a tensor with audio data
        # [channels, samples] = [2, 44000]
        if waveform.shape[0] > 1:
            # converts stereo audio into mono by averaging channel values
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform

        return spectogram, row['label']


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60*60*3)
def train():
    esc50_dir = Path("/opt/esc50-data")

    train_transform = nn.Sequential(T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_max=11025,
        f_min=0
    ),
        T.AmplitudeToDB(),
        T.TimeMasking(time_mask_param=80),
        T.FrequencyMasking(freq_mask_param=30))

    val_transform = nn.Sequential(T.MelSpectrogram(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_max=11025,
        f_min=0
    ),
        T.AmplitudeToDB())

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="train", transform=train_transform)
    val_dataset = ESC50Dataset(
        data_dir=esc50_dir, metadata_file=esc50_dir / "meta" / "esc50.csv", split="val", transform=val_transform)

    print(f"training dataset {len(train_dataset)}")
    print(f"val dataset {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    


@app.local_entrypoint()
def main():
    train.remote()
