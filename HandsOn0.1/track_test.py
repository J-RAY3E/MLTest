import time
from collections import deque
import numpy as np
import mss
import mediapipe as mp
import cv2
from tensorflow.keras.models import load_model

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sct = mss.mss()
monitor = {"top": 140, "left": 20, "width": 1000, "height": 600}

current_seq_taken = []


HAND_LM_SIZE = 21 * 3
POSE_KEYPOINTS = [11, 12, 0]
FACE_KEYPOINTS = [1, 13, 14, 33, 263, 10, 152]

labels =  ['age',
 'bad',
 'bathroom',
 'busy',
 'chat',
 'come',
 'deaf',
 'dontlike',
 'dontunderstand',
 'favorite',
 'fine',
 'friend',
 'go',
 'god',
 'goodbye',
 'great',
 'hardofhearing',
 'have',
 'havent',
 'hearing',
 'hello',
 'howareu',
 'im',
 'learn',
 'learning',
 'like',
 'myname',
 'name',
 'nicetomeetu',
 'nicetomeetyou',
 'nicetometyou',
 'no',
 'noise',
 'notalot',
 'nothing',
 'oisee',
 'ok',
 'please',
 'restaurant',
 'school',
 'seeulater',
 'seeyoulater',
 'singlanguaje',
 'slow',
 'sorry',
 'soso',
 'takecare',
 'teacher',
 'thanku',
 'understand',
 'want',
 'what',
 'whatsup',
 'where',
 'work',
 'yes',
 'you',
 'youname']

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




def positional_encoding(length, dim):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(dim))
    encoding = pos * angle_rates
    encoding[:, 0::2] = np.sin(encoding[:, 0::2])
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])
    return encoding

def parsestack(sequence):
    features = sequence[0].keys()
    end_sequence = []
    for frame in sequence:
        for feature in features:
            end_sequence.extend(frame[feature])
    return np.array(end_sequence)

class Model():
    def __init__(self, path, labels, threshold=0.8):
        self.model = load_model(path)
        self.labels = labels
        self.threshold = threshold

    def inference(self, x):
        assert x.shape == (15, 156), f"Shape incorrecto: {x.shape}"
        x = np.expand_dims(x, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        idx = np.argmax(probs)
        conf = probs[idx]
        return (self.labels[idx], conf) if conf >= self.threshold else ("none", conf)

class LockStack:
    def __init__(self, counts):
        self.target = counts
        self.counter = 0

    def verify(self):
        """Devuelve True si debe seguir contando, False si espera inicio."""
        return self.counter > 0

    def start(self):
        """Inicia la captura."""
        self.counter = 1

    def tick(self):
        """Aumenta contador, devuelve True si llegó al final."""
        if self.counter > 0:
            self.counter += 1
            if self.counter > self.target:
                return True
        return False

    def reset(self):
        self.counter = 0


def track_screen_live():
    global current_seq_taken
    current_seq_taken = []
    stack_verifier = LockStack(15*10)

    model = Model("./model_test/bilstm_sign_model.keras", labels, 0.1)

    with mp_holistic.Holistic() as holistic:
        while True:
            frame = np.array(sct.grab(monitor))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            results = holistic.process(frame_rgb)
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("a"):  # Iniciar captura
                stack_verifier.start()
                print("🎥 Capturando secuencia...")
            elif key == ord("w"):  # Limpiar buffer
                current_seq_taken.clear()
                stack_verifier.reset()
                print("♻️ Buffer limpiado.")

            if stack_verifier.verify():
                lm_dict = extract_landmarks(results)
                current_seq_taken.append(lm_dict)
                print(stack_verifier.counter,len(current_seq_taken))
                if stack_verifier.tick():  # Llegamos a 15 frames
                    all_dect(current_seq_taken,model)
                    current_seq_taken.clear()
                    stack_verifier.reset()  # Reinicia para seguir capturando

            # Dibuja landmarks en pantalla
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            cv2.imshow("Monitor capture", image)

    cv2.destroyAllWindows()

WINDOW_SIZE = 15
STRIDE = 1
def all_dect(sequence,model):
    detecciones = 0
    for start_idx in range(0, len(sequence) - WINDOW_SIZE + 1, STRIDE):
        window_seq = sequence[start_idx:start_idx + WINDOW_SIZE]
        x = parsestack(window_seq).reshape(WINDOW_SIZE, -1)
        pred, conf = model.inference(x)
        if pred != "none" and conf>.97:
            print(f"✅ Señal detectada en ventana {start_idx}-{start_idx + WINDOW_SIZE}: {pred} ({conf:.2f})")
            detecciones += 1

    print(f"Total detecciones: {detecciones}")


if __name__ == "__main__":
    track_screen_live()
