import torch
import cv2
from torchvision import transforms
from models.iris_net import IrisNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = IrisNet().to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_embedding(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img)

    return embedding.squeeze().cpu().numpy()
