import torch.nn as nn

class EmbeddingClassifier(nn.Module):
    def __init__(self, input_size=768, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_size, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)