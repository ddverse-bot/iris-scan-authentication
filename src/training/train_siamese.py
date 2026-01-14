import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.siamese_net import SiameseIrisNet
from src.training.dataset import IrisPairDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def contrastive_loss(out1, out2, label, margin=1.0):
    distance = F.pairwise_distance(out1, out2)
    loss = label * torch.pow(distance, 2) + \
           (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss.mean()

dataset = IrisPairDataset("data/raw")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SiameseIrisNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    total_loss = 0
    for img1, img2, label in loader:
        img1, img2 = img1.to(device), img2.to(device)
        label = label.float().to(device)

        out1, out2 = model(img1, img2)
        loss = contrastive_loss(out1, out2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/siamese_iris.pth")
print("✅ Siamese model trained and saved")
