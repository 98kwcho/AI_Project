# -*- coding: utf-8 -*-
import cv2
import time
import math
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List, Tuple
from openvino.runtime import Core, CompiledModel

# =========================
# 설정
# =========================
MODEL_XML = "workspace/open_model_zoo/demos/human_pose_estimation_demo/python/intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"
DEVICE = "AUTO"          # "CPU", "GPU", "AUTO" 등
CAM_INDEX = 4           # 카메라 인덱스

CONF_KPT = 0.2           # 전신 키포인트 표시 임계값
CONF_WRIST = 0.3         # 손목 박스 생성 임계값
MAX_HANDS = 4            # MediaPipe 최대 손 수

# 미러(셀피) 프리뷰 보정: 화면이 좌우반전되어 보일 때 True
MIRROR = True

# 사용 중인 모델 기준 손목 인덱스(질문 내용 반영: Left=10, Right=9)
BODY_WRIST_IDX = {'Left': 10, 'Right': 9}

# (선택) 어깨 인덱스: 몸 중심선 힌트용(모델에 맞게 수정 가능)
BODY_SHOULDER_IDX = {'Left': 5, 'Right': 2}

# =========================
# 스켈레톤 연결(요청한 순서로 고정)
# =========================
POSE_PAIRS = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
)

# =========================
# 유틸
# =========================
@dataclass
class Keypoint:
    x: float
    y: float
    conf: float

Keypoints = List[Keypoint]


# --------- 크로마키(검은 배경 투명화) ----------
def make_black_transparent(img_rgba, thr=30, soft_edge=5):
    """
    img_rgba: (H,W,3|4) BGR(A)
    thr: 회색값 <= thr 인 어두운(검은) 영역을 투명 처리
    soft_edge>0: 경계 페더링
    """
    assert img_rgba.shape[2] in (3, 4)
    if img_rgba.shape[2] == 3:
        a = np.full(img_rgba.shape[:2] + (1,), 255, np.uint8)
        img_rgba = np.concatenate([img_rgba, a], axis=2)

    bgr = img_rgba[..., :3]
    alpha = img_rgba[..., 3].copy()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    black_mask = (gray <= thr).astype(np.uint8) * 255  # 검은 영역(투명으로 만들 대상)
    if soft_edge > 0:
        inv = 255 - black_mask
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
        dist = np.clip(dist / float(soft_edge), 0.0, 1.0)
        feather = (dist * 255).astype(np.uint8)  # 검은 영역 주변을 부드럽게 0→255
        new_alpha = np.minimum(alpha, feather)   # 경계쪽 투명도 서서히 증가
    else:
        new_alpha = alpha
        new_alpha[black_mask == 255] = 0

    out = img_rgba.copy()
    out[..., 3] = new_alpha
    return out

# ---------------------------------------------

# --------- 자동 트리밍(알파 채널 기준) ----------
def trim_transparent_area(img_rgba):
    """알파 채널이 0이 아닌 부분만 잘라내기 (자동 트리밍)"""
    if img_rgba is None or img_rgba.shape[2] < 4:
        return img_rgba
    alpha = img_rgba[..., 3]
    ys, xs = np.where(alpha > 10)
    if len(ys) == 0 or len(xs) == 0:
        return img_rgba
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return img_rgba[y1:y2+1, x1:x2+1]


def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)

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
        out = output.squeeze(0) if output.ndim == 4 else output  # [C,H,W]
        C, Hh, Wh = out.shape
        max_k = min(18, C)  # 0~17 관절만 사용
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
# 손 추정(미디어파이프) 모듈
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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)
    def draw(self, frame_bgr: np.ndarray, results):
        if not results.multi_hand_landmarks:
            return []
        drawn = []
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            color = (0,255,0) if label == "Left" else (255,0,0)
            self.mp_drawing.draw_landmarks(
                frame_bgr, hand_lm, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(thickness=2)
            )
            drawn.append((label, handedness.classification[0].score, hand_lm))
        return drawn

# =========================
# 시각화(스켈레톤 & 손-손목 연결)
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
        if not body_kpts:
            return
        # 전신 손목 좌표 수집
        wrists = {}
        for side, idx in BODY_WRIST_IDX.items():
            if idx < len(body_kpts) and body_kpts[idx].conf > self.conf_thr:
                wrists[side] = (int(body_kpts[idx].x), int(body_kpts[idx].y))

        # 몸 중심선(어깨 평균 x)
        mid_x = None
        ls, rs = BODY_SHOULDER_IDX['Left'], BODY_SHOULDER_IDX['Right']
        if ls < len(body_kpts) and rs < len(body_kpts):
            if body_kpts[ls].conf > self.conf_thr and body_kpts[rs].conf > self.conf_thr:
                mid_x = 0.5 * (body_kpts[ls].x + body_kpts[rs].x)

        d_thr = max(img_w, img_h) * 0.2  # 최대 허용 거리

        for label, score, hand_lm in hands_drawn:
            hlabel = ('Right' if label == 'Left' else 'Left') if MIRROR else label
            w0 = hand_lm.landmark[0]
            hx, hy = int(w0.x * img_w), int(w0.y * img_h)

            best_pt, best_d2 = None, 1e18
            # 1) 라벨 일치 우선
            if hlabel in wrists:
                wx, wy = wrists[hlabel]
                d2 = (wx - hx)**2 + (wy - hy)**2
                best_pt, best_d2 = (wx, wy), d2
            # 2) 폴백: 다른쪽
            if best_pt is None and wrists:
                for _, (wx, wy) in wrists.items():
                    d2 = (wx - hx)**2 + (wy - hy)**2
                    if d2 < best_d2:
                        best_pt, best_d2 = (wx, wy), d2
            # 3) 자연스러운 쪽(중심선) 우대
            if best_pt is not None and mid_x is not None and hlabel in wrists:
                natural = 'Left' if hx < mid_x else 'Right'
                if natural in wrists and natural != hlabel:
                    wx2, wy2 = wrists[natural]
                    d2_nat = (wx2 - hx)**2 + (wy2 - hy)**2
                    if d2_nat * 0.8 < best_d2:
                        best_pt, best_d2 = (wx2, wy2), d2_nat
                        hlabel = natural
            # 4) 거리 임계값
            if best_pt is not None and best_d2 <= (d_thr**2):
                cv2.line(img, (hx, hy), best_pt, (255, 255, 255), 2)
                disp = (label if not MIRROR else ('Left' if label == 'Right' else 'Right'))
                cv2.putText(img, f"{disp}", (hx - 30, hy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0,255,0) if disp == "Left" else (255,0,0), 2)
                
    def overlay_png(img, png, center_x, center_y):
        ph, pw = png.shape[:2]

        # 좌표 계산 (PNG를 중심에 놓는다)
        x1 = int(center_x - pw // 2)
        y1 = int(center_y - ph // 2)
        x2 = x1 + pw
        y2 = y1 + ph

        # 화면 범위를 벗어나지 않도록
        h, w = img.shape[:2]
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return img  # 범위 벗어나면 스킵

        # PNG split (B,G,R,A)
        b,g,r,a = cv2.split(png)
        overlay = cv2.merge((b,g,r))
        mask = cv2.merge((a,a,a)) / 255.0

        # 오버레이 영역
        roi = img[y1:y2, x1:x2]

        # 합성
        img[y1:y2, x1:x2] = (roi * (1-mask) + overlay * mask).astype(np.uint8)
        return img


# =========================
# 손 ROI(선택 기능)
# =========================
def get_hand_regions_from_wrist(kpts: Keypoints, img_w: int, img_h: int, size: int = 128) -> List[Tuple[int,int,int,int]]:
    rois = []
    for idx in BODY_WRIST_IDX.values():
        if idx < len(kpts) and kpts[idx].conf > CONF_WRIST:
            x, y = int(kpts[idx].x), int(kpts[idx].y)
            rois.append(clamp_box(x - size//2, y - size//2, x + size//2, y + size//2, img_w, img_h))
    return rois


# =========================
# PNG 오버레이 안전 버전 이미지 아래쪽을 붙이는 코드
# =========================

def overlay_png(img, png, center_x, center_y):
    """
    img: 배경 (BGR)
    png: 오버레이할 PNG (BGRA)
    center_x, center_y: PNG의 하단 중앙이 맞춰질 기준점 (예: 오른손 5번 좌표)
    """
    ph, pw = png.shape[:2]

    # ===== PNG 하단 중앙을 손 좌표에 맞춤 =====
    x1 = int(center_x - pw / 2)
    y2 = int(center_y)             # 손 좌표가 PNG의 하단
    x2 = x1 + pw
    y1 = int(y2 - ph)              # PNG 높이만큼 위로 이동
    # ==========================================

    # ===== 화면 범위 보정 =====
    h, w = img.shape[:2]
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return img

    # 안전한 클램프 (화면을 벗어나는 부분 자르기)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)

    roi = img[y1c:y2c, x1c:x2c]
    png_crop = png[(y1c - y1):(y2c - y1), (x1c - x1):(x2c - x1)]

    # ===== BGRA 분리 및 마스크 처리 =====
    if png_crop.shape[2] == 4:
        b, g, r, a = cv2.split(png_crop)
        overlay = cv2.merge((b, g, r))
        mask = cv2.merge((a, a, a)) / 255.0
    else:
        overlay = png_crop
        mask = np.ones_like(overlay, dtype=np.float32)

    # ===== 합성 =====
    img[y1c:y2c, x1c:x2c] = (roi * (1 - mask) + overlay * mask).astype(np.uint8)

    return img







# =========================
# 메인
# =========================
def main():
    pose = OpenVinoPose(MODEL_XML, DEVICE)
    hands = MediaPipeHands(MAX_HANDS)

    # ===============================
    # 53장의 PNG 프레임 로드
    # ===============================
    import glob
    frame_files = sorted(glob.glob("datasets/LightsaberPNG/ezgif-frame-*.png"))
    if not frame_files:
        raise FileNotFoundError("⚠️ PNG 프레임 파일을 찾을 수 없습니다.")

    print(f"🔹 {len(frame_files)}개의 PNG 프레임을 로드했습니다.")

    overlay_frames = []
    for f in frame_files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        img = make_black_transparent(img, thr=30, soft_edge=5)
        overlay_frames.append(img)

    total_frames = len(overlay_frames)
    frame_idx = 0  # 현재 오버레이 프레임 인덱스

    # 🔧 이미지 스케일 조절 관련 변수 (여기에 추가)
    scale = 1.0
    SCALE_STEP = 0.1
    MIN_SCALE  = 0.25
    MAX_SCALE  = 3.0


    cap = cv2.VideoCapture(CAM_INDEX)
    prev_t = 0.0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        # ============= 전신 포즈 추론 =============
        out = pose.infer(frame)
        body_kpts = pose.extract_keypoints(out, w, h)
        viz = Visualizer()
        viz.draw_skeleton(frame, body_kpts, (0,255,0))

        # ============= 손 추정 (MediaPipe) =========
        results = hands.process(frame)
        hands_drawn = hands.draw(frame, results)

        # MIRROR 고려한 오른손 판별
        right_hand_kpts = None
        for label, score, hand_lm in hands_drawn:
            is_right = (label == "Right") if not MIRROR else (label == "Left")
            if is_right:
                right_hand_kpts = [Keypoint(lm.x * w, lm.y * h, 1.0) for lm in hand_lm.landmark]
                break

        # ============= 53장 중 현재 프레임 표시 =============
        if right_hand_kpts and len(right_hand_kpts) > 5:
            x, y = int(right_hand_kpts[5].x), int(right_hand_kpts[5].y)

            # 🔧 현재 프레임 가져오기
            base = overlay_frames[frame_idx]

            # 🔧 스케일 조절
            if scale != 1.0:
                new_w = max(1, int(base.shape[1] * scale))
                new_h = max(1, int(base.shape[0] * scale))
                overlay_img = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                overlay_img = base

            # 🔧 오버레이
            frame = overlay_png(frame, overlay_img, x, y)
            

            # 다음 프레임으로 이동 (53장 반복)
            frame_idx = (frame_idx + 1) % total_frames

        # ============= FPS 표시 =============
        now = time.time()
        fps = 1.0 / (now - prev_t) if prev_t else 0.0
        prev_t = now
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        # ============= 출력 =============
        cv2.imshow("Full Body + Hand Pose + Lightsaber Animation", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC 종료
            break
        elif key in (ord('-'), ord('_')):  # 🔻 작게
            scale = max(MIN_SCALE, round(scale - SCALE_STEP, 2))
            print(f"🔻 이미지 축소: scale={scale}")
        elif key in (ord('='), ord('+')):  # 🔺 크게
            scale = min(MAX_SCALE, round(scale + SCALE_STEP, 2))
            print(f"🔺 이미지 확대: scale={scale}")


    cap.release()
    cv2.destroyAllWindows()





if __name__ == "__main__":
    main()
