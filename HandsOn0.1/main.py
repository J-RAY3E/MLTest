import numpy as np
import mss
import mediapipe as mp
import cv2
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sct = mss.mss()
monitor = {"top": 140, "left": 20, "width": 1000, "height": 600}

current_seq_taken = []
all_sequences = []

HAND_LM_SIZE = 21 * 3
POSE_KEYPOINTS = [11, 12, 0]
FACE_KEYPOINTS = [1, 13, 14, 33, 263, 10, 152]

def extract_landmarks(results):
    data = {}

    if results.left_hand_landmarks:
        left = []
        for lm in results.left_hand_landmarks.landmark:
            left.extend([lm.x, lm.y, lm.z])
        data["left_hand"] = left
    else:
        data["left_hand"] = [0.0] * HAND_LM_SIZE


    if results.right_hand_landmarks:
        right = []
        for lm in results.right_hand_landmarks.landmark:
            right.extend([lm.x, lm.y, lm.z])
        data["right_hand"] = right
    else:
        data["right_hand"] = [0.0] * HAND_LM_SIZE

    if results.pose_landmarks:
        pose = []
        for i in POSE_KEYPOINTS:
            lm = results.pose_landmarks.landmark[i]
            pose.extend([lm.x, lm.y, lm.z])
        data["pose"] = pose
    else:
        data["pose"] = [0.0] * (len(POSE_KEYPOINTS) * 3)

    if results.face_landmarks:
        face = []
        for i in FACE_KEYPOINTS:
            lm = results.face_landmarks.landmark[i]
            face.extend([lm.x, lm.y, lm.z])
        data["face"] = face
    else:
        data["face"] = [0.0] * (len(FACE_KEYPOINTS) * 3)

    return data


def track_image():
    global  current_seq_taken
    with mp_holistic.Holistic() as holistic:
        stack_verifier = LockStack(15)
        while True:
            frame = np.array(sct.grab(monitor))
            frame.flags.writeable = False
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            frame.flags.writeable = True
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if stack_verifier.verify():
                lm_dict = extract_landmarks(results)
                current_seq_taken.append(lm_dict)
                if stack_verifier.isLast():
                    print("Etiqueta para la secuencia:")
                    label = input()
                    if "el" != label:
                        all_sequences.append({"label": label, "sequence": current_seq_taken})
                        current_seq_taken = []
                        print(f"Secuencia guardada con etiqueta: {label}")
                    else:
                        current_seq_taken = []
                        print("secuencia no guardada")

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.imshow("Picking image", image)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break

            elif key == ord("q"):
                stack_verifier.start()

            elif key == ord("w"):
                print("Limpiando caché de secuencia...")
                current_seq_taken.clear()


class LockStack():
    def __init__(self,counts):
        self.counts = counts
        self.current = counts
    def verify(self):
        if self.counts == self.current:
            return False
        self.counts += 1
        return True
    def start(self):
        self.counts = 0

    def isLast(self):
        return self.current == self.counts

if __name__ == "__main__":
    track_image()
    all_sequences = pd.DataFrame(all_sequences)
    all_sequences.to_csv("data_fetched5.csv")

