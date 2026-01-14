# Iris Scan Authentication using Siamese Networks

An end-to-end biometric iris authentication system built using PyTorch, Siamese neural networks, and webcam-based image capture.
The system supports user enrollment, template storage, and distance-based verification for secure identity authentication.

 Motivation

Traditional password-based authentication systems suffer from:

Weak security

Password reuse

Easy spoofing

Biometric authentication, especially iris recognition, offers:

High uniqueness

Long-term stability

Strong resistance to impersonation

This project implements a deep learning–based iris verification system using metric learning.

 Core Idea

Instead of classifying users directly, we use a Siamese Network to learn an embedding space where:

Iris images from the same person are close

Iris images from different people are far apart

Authentication is done by comparing embedding distances.

 System Architecture
Webcam → Iris Image → CNN Encoder → Embedding Vector
                                     ↓
                            Stored User Templates
                                     ↓
                           Distance Comparison
                                     ↓
                           Access Granted / Denied

 Project Structure
iris_scan_project/
├── src/
│   ├── capture/          # Webcam image capture
│   ├── preprocessing/   # Image preprocessing
│   ├── inference/       # Embedding extraction
│   └── training/        # Siamese network & dataset
│
├── scripts/
│   ├── enroll_user.py   # User enrollment
│   ├── verify_user.py   # User authentication
│
├── data/
│   ├── raw/             # Raw iris images (ignored in Git)
│   ├── templates/       # Stored embeddings (ignored in Git)
│
├── models/
│   └── siamese_iris.pth # Trained model (ignored in Git)
│
├── requirements.txt
├── .gitignore
└── README.md


 Note: Raw biometric data and templates are intentionally excluded from Git for privacy and ethical reasons.

Tech Stack

Python 3

PyTorch

Torchvision

OpenCV

NumPy

Pillow

 Model Details

Architecture: Siamese Network

Backbone: ResNet-18 (pretrained)

Loss Function: Contrastive Loss

Distance Metric: Euclidean Distance

Output: Fixed-length iris embedding vector

 Installation
 Clone the repository
git clone https://github.com/ddverse-bot/iris-scan-authentication.git
cd iris-scan-authentication

2️ Install dependencies
pip install -r requirements.txt

 Training the Siamese Network

Ensure your dataset follows this structure:

data/raw/
├── user1/
│   ├── img1.png
│   ├── img2.png
├── user2/
│   ├── img1.png
│   ├── img2.png


Each user must have at least 2 images.

Run training:

python -m src.training.train_siamese


Trained model will be saved to:

models/siamese_iris.pth

 User Enrollment

Enroll a new user by capturing iris images via webcam and storing embeddings.

python -m scripts.enroll_user


Output:

data/templates/<user_id>.pkl


Each user can have multiple enrollment samples.

 User Verification

Verify a user by capturing a live iris image and comparing it with stored templates.

python -m scripts.verify_user


Decision is based on a distance threshold:

Distance < threshold →  Access Granted

Distance ≥ threshold →  Access Denied

 Evaluation (Planned)

Future improvements include:

FAR (False Accept Rate)

FRR (False Reject Rate)

ROC curve analysis

Threshold optimization

Security & Ethics

No biometric data is uploaded to GitHub

Templates are stored locally

System designed for educational & research purposes only