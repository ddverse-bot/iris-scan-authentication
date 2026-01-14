import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class IrisPairDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        # Keep only users with at least 2 images
        self.images = {}
        for user in os.listdir(root_dir):
            user_path = os.path.join(root_dir, user)
            if os.path.isdir(user_path):
                imgs = os.listdir(user_path)
                if len(imgs) >= 2:
                    self.images[user] = imgs

        self.users = list(self.images.keys())

        if len(self.users) < 2:
            raise ValueError("Dataset needs at least 2 users with 2 images each")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return 10000  # virtual size

    def __getitem__(self, idx):
        same = random.choice([0, 1])

        if same:
            user = random.choice(self.users)
            img1_name, img2_name = random.sample(self.images[user], 2)

            img1_path = os.path.join(self.root_dir, user, img1_name)
            img2_path = os.path.join(self.root_dir, user, img2_name)
            label = 1.0

        else:
            user1, user2 = random.sample(self.users, 2)
            img1_name = random.choice(self.images[user1])
            img2_name = random.choice(self.images[user2])

            img1_path = os.path.join(self.root_dir, user1, img1_name)
            img2_path = os.path.join(self.root_dir, user2, img2_name)
            label = 0.0

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        return self.transform(img1), self.transform(img2), label
