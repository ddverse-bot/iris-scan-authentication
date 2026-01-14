import pickle
from src.capture.webcam_capture import capture_image
from src.inference.embed import get_embedding
from src.utils.similarity import cosine_similarity

user_id = input("Enter User ID to verify: ")

image_path = capture_image("temp")
if image_path is None:
    print("Verification cancelled")
    exit()

test_embedding = get_embedding(image_path)

with open(f"data/templates/{user_id}.pkl", "rb") as f:
    stored_embedding = pickle.load(f)

score = cosine_similarity(stored_embedding, test_embedding)
print("Similarity score:", score)

if score > 0.85:
    print("✅ AUTHENTICATION SUCCESS")
else:
    print("❌ AUTHENTICATION FAILED")
