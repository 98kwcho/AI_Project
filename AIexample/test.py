# -*- coding: utf-8 -*-
import cv2
import time
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple
from openvino.runtime import Core, CompiledModel

# =========================
# 설정
# =========================
MODEL_XML = "intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"
DEVICE = "AUTO"
CAM_INDEX = 4

CONF_KPT = 0.2
CONF_WRIST = 0.3
MAX_HANDS = 4
MIRROR = True

BODY_WRIST_IDX = {'Left': 10, 'Right': 9}
BODY_SHOULDER_IDX = {'Left': 5, 'Right': 2}

POSE_PAIRS = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
)

# =========================
# PNG 이미지 로드
# =========================
PNG_IMG = cv2.imread("rock.png", cv2.IMREAD_UNCHANGED)
if PNG_IMG is None:
    print("[ERROR] hand_icon.png 파일을 찾을 수 없습니다.")
    exit()
PNG_IMG = cv2.resize(PNG_IMG, (100,100))  # 크기 조절 가능

# =========================
# 유틸
# =========================
@dataclass
class Keypoint:
    x: float
    y: float
    conf: float

Keypoints = List[Keypoint]

def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)

# ✅ PNG 오버레이 함수
def overlay_png(img, png, cx, cy):
    ph, pw = png.shape[:2]
    x1 = int(cx - pw // 2)
    y1 = int(cy - ph // 2)
    x2 = x1 + pw
    y2 = y1 + ph

    h, w = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return img

    b,g,r,a = cv2.split(png)
    overlay = cv2.merge((b,g,r))
    mask = cv2.merge((a,a,a)) / 255.0
    roi = img[y1:y2, x1:x2]

    img[y1:y2, x1:x2] = (roi * (1-mask) + overlay * mask).astype(np.uint8)
    return img

# =========================
# 전신 포즈 추정 모듈(OpenVINO)
# =========================
class OpenVinoPose:
    def __init__(self, model_xml: str, device: str = "AUTO"):
        ie = Core()
        model = ie.read_model(model_xml)
        self.compiled: CompiledModel = ie.compile_model(model, device)
        self.input_port = self.compiled.input(0)
        self.output_port = self.compiled.output(0)
        _, _, self.in_h, self.in_w = self.input_port.shape
        print(f"[Pose] Input expects: {self.in_h}x{self.in_w}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        img_resized = cv2.resize(frame, (self.in_w, self.in_h))
        img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :].astype(np.float32)
        return img_input

    def infer(self, frame: np.ndarray) -> np.ndarray:
        inp = self.preprocess(frame)
        res = self.compiled({self.input_port: inp})[self.output_port]
        return np.asarray(res)

    def extract_keypoints(self, output: np.ndarray, orig_w: int, orig_h: int) -> Keypoints:
        out = output.squeeze(0)
        C, Hh, Wh = out.shape
        max_k = min(18, C)
        kpts: Keypoints = []
        for i in range(max_k):
            hm = out[i]
            _, conf, _, point = cv2.minMaxLoc(hm)
            x_hm, y_hm = point
            x = int(x_hm * orig_w / Wh)
            y = int(y_hm * orig_h / Hh)
            kpts.append(Keypoint(x, y, float(conf)))
        return kpts

# =========================
# 미디어파이프 손
# =========================
class MediaPipeHands:
    def __init__(self, max_hands: int = 2):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=max_hands
        )

    def process(self, frame_bgr: np.ndarray):
        return self.hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def draw(self, frame_bgr: np.ndarray, results):
        if not results.multi_hand_landmarks:
            return []
        drawn = []
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            self.mp_drawing.draw_landmarks(frame_bgr, hand_lm, self.mp_hands.HAND_CONNECTIONS)
            drawn.append((label, handedness.classification[0].score, hand_lm))
        return drawn

# =========================
# 스켈레톤 & PNG 붙이기
# =========================
class Visualizer:
    def __init__(self, pose_pairs=POSE_PAIRS, conf_thr=CONF_KPT):
        self.pose_pairs = pose_pairs
        self.conf_thr = conf_thr

    def draw_keypoints(self, img: np.ndarray, kpts: Keypoints, color=(0,255,0)):
        for kp in kpts:
            if kp.conf > self.conf_thr:
                cv2.circle(img, (int(kp.x), int(kp.y)), 3, color, -1)

    def draw_skeleton(self, img: np.ndarray, kpts: Keypoints, color=(0,255,0)):
        self.draw_keypoints(img, kpts, color)
        for a, b in self.pose_pairs:
            if a < len(kpts) and b < len(kpts):
                ka, kb = kpts[a], kpts[b]
                if ka.conf > self.conf_thr and kb.conf > self.conf_thr:
                    cv2.line(img, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), color, 2)

    def draw_hand_labels_and_attach(self, img: np.ndarray, hands_drawn, img_w, img_h, body_kpts: Keypoints):
        for label, score, hand_lm in hands_drawn:

            # 손바닥 중심 landmark[0]
            palm = hand_lm.landmark[8]  # 손가락 끝(검지)
            cx, cy = int(palm.x * img_w), int(palm.y * img_h)

            # PNG 붙이기
            overlay_png(img, PNG_IMG, cx, cy)

# =========================
# 메인
# =========================
def main():
    pose = OpenVinoPose(MODEL_XML, DEVICE)
    hands = MediaPipeHands(MAX_HANDS)
    viz = Visualizer()

    cap = cv2.VideoCapture(CAM_INDEX)
    prev_t = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]

        out = pose.infer(frame)
        body_kpts = pose.extract_keypoints(out, w, h)
        viz.draw_skeleton(frame, body_kpts)

        results = hands.process(frame)
        hands_drawn = hands.draw(frame, results)
        viz.draw_hand_labels_and_attach(frame, hands_drawn, w, h, body_kpts)

        now = time.time()
        fps = 1.0 / (now - prev_t) if prev_t else 0.0
        prev_t = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Full Body + Hand PNG", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
