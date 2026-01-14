import pickle
import os
import sys
sys.path.append(".")

from src.capture.webcam_capture import capture_image
from src.inference.embed import get_embedding

user_id = input("Enter new User ID: ")

image_path = capture_image(user_id)
if image_path is None:
    print("Enrollment cancelled")
    exit()

embedding = get_embedding(image_path)

os.makedirs("data/templates", exist_ok=True)

template_path = f"data/templates/{user_id}.pkl"

# Load existing embeddings if user already exists
if os.path.exists(template_path):
    with open(template_path, "rb") as f:
        embeddings = pickle.load(f)
else:
    embeddings = []

embeddings.append(embedding)

with open(template_path, "wb") as f:
    pickle.dump(embeddings, f)

print("✅ User enrolled successfully")
print(f"Total samples for {user_id}: {len(embeddings)}")
