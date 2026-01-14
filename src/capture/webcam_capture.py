import cv2
import os

def capture_image(user_id, save_dir="data/raw"):
    os.makedirs(f"{save_dir}/{user_id}", exist_ok=True)

    cap = cv2.VideoCapture(0)
    print("Press 's' to save image | 'q' to quit")

    while True:
        ret, frame = cap.read()
        cv2.imshow("Iris Capture", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            path = f"{save_dir}/{user_id}/iris.png"
            cv2.imwrite(path, frame)
            print("Saved:", path)
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return path
