import torch
import torch.nn as nn
import glob
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class PrecomputedFeatureDataset(Dataset):
    def __init__(self, features_dir):
        self.files = glob.glob(os.path.join(features_dir, '**', '*.pt'), recursive=True)
        self.genres = sorted(['blues', 'classical', 'country', 'disco', 'hiphop',
                              'jazz', 'metal', 'pop', 'reggae', 'rock'])
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        genre = Path(file_path).parent.name
        label = self.genre_to_idx[genre]
        feature = torch.load(file_path)
        # Convert stereo to mono by averaging channels
        if feature.shape[0] > 1:
            feature = feature.mean(dim=0, keepdim=True)
        return feature, label


class CRNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # CNN backbone: input shape (Batch, 1, 128, Time)
        self.cnn = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Bidirectional LSTM
        # After two MaxPool2d(2): Mels = 128/4 = 32, Channels = 64
        # Input size to LSTM = 64 * 32 = 2048
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Classifier: bidirectional doubles the hidden size
        self.fc = nn.Linear(64 * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, 128, Time)

        # Pass through CNN
        x = self.cnn(x)
        # Shape: (Batch, 64, 32, Time//4)

        # Reshape for LSTM: (Batch, Time, Channels*Mels)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)       # (Batch, Time, Channels, Mels)
        x = x.reshape(b, t, c * f)       # (Batch, Time, 2048)

        # Pass through LSTM
        x, _ = self.lstm(x)
        # Shape: (Batch, Time, 128) since bidirectional

        # Global max pooling over time
        x, _ = torch.max(x, dim=1)
        # Shape: (Batch, 128)

        # Classifier
        logits = self.fc(x)
        return logits