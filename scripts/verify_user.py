import pickle
import sys
import os
import numpy as np

sys.path.append(".")

from src.capture.webcam_capture import capture_image
from src.inference.embed import get_embedding

user_id = input("Enter User ID to verify: ")

template_path = f"data/templates/{user_id}.pkl"
if not os.path.exists(template_path):
    print("❌ User not enrolled")
    exit()

image_path = capture_image("verify")
if image_path is None:
    print("Verification cancelled")
    exit()

query_embedding = get_embedding(image_path)

with open(template_path, "rb") as f:
    stored_embeddings = pickle.load(f)

distances = [
    np.linalg.norm(query_embedding - emb)
    for emb in stored_embeddings
]

min_dist = min(distances)
THRESHOLD = 0.8  # tune later

print(f"Min distance: {min_dist:.4f}")

if min_dist < THRESHOLD:
    print("✅ ACCESS GRANTED")
else:
    print("❌ ACCESS DENIED")
