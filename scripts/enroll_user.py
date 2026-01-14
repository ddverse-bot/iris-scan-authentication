import pickle
import os
from src.capture.webcam_capture import capture_image
from src.inference.embed import get_embedding

user_id = input("Enter new User ID: ")

image_path = capture_image(user_id)
if image_path is None:
    print("Enrollment cancelled")
    exit()

embedding = get_embedding(image_path)

os.makedirs("data/templates", exist_ok=True)

with open(f"data/templates/{user_id}.pkl", "wb") as f:
    pickle.dump(embedding, f)

print("✅ User enrolled successfully")
