import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class TinySiamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((8,8))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 256), nn.ReLU(),
        )

    def embed(self, x):
        x = self.conv(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, xa, xb):
        ea = self.embed(xa)
        eb = self.embed(xb)
        return ea, eb

_model = None

def _load_model():
    global _model
    if _model is None:
        _model = TinySiamese().eval()
    return _model

def siamese_compare(imgA, imgB):
    model = _load_model()
    def prep(img):
        img = cv2.resize(img, (256,256))
        t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        return t
    ta = prep(imgA)
    tb = prep(imgB)
    with torch.no_grad():
        ea, eb = model(ta, tb)
        sim = F.cosine_similarity(ea, eb).item()
    return sim
