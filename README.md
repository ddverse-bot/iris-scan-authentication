#  Iris Scan Authentication System (PyTorch)

A **deep learning–based iris biometric authentication system** built using **PyTorch and OpenCV**.  
This project captures iris images via a webcam, converts them into **secure numerical embeddings**, and verifies identity using **cosine similarity**.

---

##  Project Overview

Biometric systems provide a secure alternative to passwords.  
Among biometrics, **iris patterns are highly unique, stable, and difficult to forge**.

This project implements a **modular, scalable MVP** for iris-based authentication using:
- Convolutional Neural Networks (CNNs)
- Real-time image capture
- Embedding-based verification

---

##  Features

-  Real-time iris image capture using webcam  
-  Deep learning feature extraction using **ResNet18 (PyTorch)**  
-  128-dimensional biometric embeddings  
-  Secure template storage (no raw images used for matching)  
-  Cosine similarity–based verification  
-  Clean, modular project structure  
-  Ready for research, internships, and competitions  

---

##  Project Structure
     iris_scan_project/
├── data/
│ ├── raw/ # Captured iris images
│ └── templates/ # Stored iris embeddings (.pkl)
├── models/
│ ├── iris_net.py # CNN model definition
│ └── init.py
├── src/
│ ├── capture/ # Webcam capture logic
│ ├── preprocessing/ # Image preprocessing
│ ├── inference/ # Embedding generation
│ ├── utils/ # Similarity functions
│ └── init.py
├── scripts/
│ ├── enroll_user.py # User enrollment
│ └── verify_user.py # User verification
├── requirements.txt
└── README.md

---

##  Tech Stack

| Component | Technology |
|--------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Vision | OpenCV |
| Model | ResNet18 |
| Similarity | Cosine Similarity |
| Hardware | Standard Webcam |

---

##  Installation & Setup

###  Clone the repository
```bash
git clone <your-repo-link>
cd iris_scan_project

### Install dependencies
```bash
pip install -r requirements.txt
How to Run

 Always run from the project root directory

- Enroll a New User
python -m scripts.enroll_user


Steps:

-Enter a user ID (e.g., user1)
-Webcam opens
-Look straight at the camera
-Press S to capture iris
-Iris embedding is saved securely

Verify a User
python -m scripts.verify_user


Steps:

Enter the same user ID

Webcam opens again

Press S

Similarity score is computed

Access is granted or denied

 Authentication Logic

Each iris scan is converted into a 128-D normalized embedding

Matching is done using cosine similarity

Threshold:

similarity > 0.85 → Authentication Success

 Security Note

Raw iris images are not used for matching

Only numerical embeddings are stored

This reduces risk of biometric data leakage

 Limitations (MVP)

Basic preprocessing (no precise iris segmentation yet)

No anti-spoofing (photo/video attack detection)

Accuracy depends on lighting and camera quality

