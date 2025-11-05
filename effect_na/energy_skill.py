# -*- coding: utf-8 -*-
import cv2
import time
import glob
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import List
from openvino.runtime import Core, CompiledModel
from PySide6.QtGui import QImage, QPixmap

# =========================
# 설정
# =========================
MODEL_XML = "intel/human-pose-estimation-0005/FP32/human-pose-estimation-0005.xml"
DEVICE = "AUTO"
CONF_KPT = 0.2
MAX_HANDS = 4
MIRROR = False

ANIM_DIR_GLOB = "hand_anim/eff_1PNG/*.png"
ANIM_SIZE = (160, 160)
ANIM_SPEED = 2
DEBOUNCE_N = 2

POSE_PAIRS = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
)

@dataclass
class Keypoint:
    x: float
    y: float
    conf: float

# =========================
# PNG 애니메이션 로드
# =========================
ANIM_FRAMES = []
for file in sorted(glob.glob(ANIM_DIR_GLOB)):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.resize(img, ANIM_SIZE)
        ANIM_FRAMES.append(img)
if not ANIM_FRAMES:
    raise RuntimeError("[ERROR] PNG 없음")

# 마스크 적용
def overlay_energy(img, energy, cx, cy):
    ph, pw = energy.shape[:2]
    x1 = int(cx - pw//2)
    y1 = int(cy - ph//2)
    x2 = x1 + pw
    y2 = y1 + ph

    h, w = img.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return img

    overlay = energy.astype(float) / 255.0
    gray = cv2.cvtColor(energy, cv2.COLOR_BGR2GRAY)
    gray = cv2.pow(gray/255.0, 0.6)
    mask = cv2.GaussianBlur(gray, (7,7),0)
    mask = np.dstack([mask,mask,mask])

    roi = img[y1:y2, x1:x2].astype(float)/255.0
    blended = roi*(1-mask) + overlay*mask
    img[y1:y2, x1:x2] = (blended*255).astype(np.uint8)
    return img

# =========================
# OpenVINO 포즈
# =========================
class OpenVinoPose:
    def __init__(self):
        ie = Core()
        model = ie.read_model(MODEL_XML)
        self.compiled: CompiledModel = ie.compile_model(model, DEVICE)
        self.input_port = self.compiled.input(0)
        self.output_port = self.compiled.output(0)
        _, _, self.in_h, self.in_w = self.input_port.shape

    def infer(self, frame):
        img = cv2.resize(frame, (self.in_w, self.in_h))
        inp = img.transpose(2,0,1)[np.newaxis,:].astype(np.float32)
        res = self.compiled({self.input_port: inp})[self.output_port]
        return np.asarray(res)

    def extract_keypoints(self, output, w, h):
        out = output.squeeze(0)
        C, Hh, Wh = out.shape
        kpts=[]
        for i in range(min(18,C)):
            hm=out[i]
            _,conf,_,pt=cv2.minMaxLoc(hm)
            x=int(pt[0]*w/Wh)
            y=int(pt[1]*h/Hh)
            kpts.append(Keypoint(x,y,float(conf)))
        return kpts

# =========================
# MediaPipe Hands
# =========================
class MediaPipeHands:
    def __init__(self):
        self.mp = mp.solutions.hands
        self.drawer = mp.solutions.drawing_utils
        self.hands = self.mp.Hands(max_num_hands=MAX_HANDS,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)
    def process(self, frame):
        return self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# palm center
def get_palm_center(lm, w, h):
    mcp=[2,5,9,13,17]
    xs=[lm.landmark[idx].x*w for idx in mcp]
    ys=[lm.landmark[idx].y*h for idx in mcp]
    return sum(xs)/5.0, sum(ys)/5.0

# fist detection
def detect_fist_ratio(lm, w, h):
    finger_pairs=[(4,2),(8,5),(12,9),(16,13),(20,17)]
    ratios=[]
    wx,wy=lm.landmark[0].x*w, lm.landmark[0].y*h
    for tip,mcp in finger_pairs:
        tx,ty=lm.landmark[tip].x*w,lm.landmark[tip].y*h
        mx,my=lm.landmark[mcp].x*w,lm.landmark[mcp].y*h
        d1=((tx-mx)**2+(ty-my)**2)**0.5
        d2=((mx-wx)**2+(my-wy)**2)**0.5
        if d2>1:
            ratios.append(d1/d2)
    if not ratios: return "unknown"
    avg=sum(ratios)/len(ratios)
    if avg<0.55: return "fist"
    if avg>0.80: return "open"
    return "unknown"

class RightHandSkill:
    def __init__(self):
        self.prev="open"
        self.active=False
        self.frame_idx=0
        self.last="unknown"
        self.streak=0

    def stable(self, label):
        if label==self.last:
            self.streak+=1
        else:
            self.last=label
            self.streak=1
        return label if self.streak>=DEBOUNCE_N else "unknown"

    def update(self,label):
        if self.prev=="fist" and label=="open":
            self.active=True
            self.frame_idx=0
        if label!="unknown":
            self.prev=label

    def draw(self,frame,lm,w,h):
        if not self.active: return
        px,py=get_palm_center(lm,w,h)
        png=ANIM_FRAMES[(self.frame_idx*ANIM_SPEED)%len(ANIM_FRAMES)]
        overlay_energy(frame,png,int(px),int(py))
        self.frame_idx+=1
        if (self.frame_idx*ANIM_SPEED)>=len(ANIM_FRAMES):
            self.active=False

# =========================
# ✅ GUI에서 호출할 함수
# =========================
def run_energy_skill(video_label, cam_index=0):
    pose = OpenVinoPose()
    hands = MediaPipeHands()
    skill = RightHandSkill()
    cap = cv2.VideoCapture(cam_index)
    prev = 0

    def update():
        nonlocal prev
        ok, frame = cap.read()
        if not ok:
            return

        if MIRROR:
            frame = cv2.flip(frame,1)

        h,w = frame.shape[:2]

        # pose skeleton
        out = pose.infer(frame)
        kpts = pose.extract_keypoints(out,w,h)
        for a,b in POSE_PAIRS:
            if a<len(kpts) and b<len(kpts):
                ka,kb = kpts[a],kpts[b]
                if ka.conf>CONF_KPT and kb.conf>CONF_KPT:
                    cv2.line(frame,(ka.x,ka.y),(kb.x,kb.y),(0,255,0),2)

        # hands
        res = hands.process(frame)
        if res.multi_hand_landmarks:
            for lm,hd in zip(res.multi_hand_landmarks,res.multi_handedness):

                # ✅ 손 랜드마크 시각화
                hands.drawer.draw_landmarks(
                    frame,
                    lm,
                    hands.mp.HAND_CONNECTIONS,
                    hands.drawer.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                    hands.drawer.DrawingSpec(color=(0,255,0), thickness=2)
                )

                # ✅ 오른손만 사용
                label = hd.classification[0].label.lower().strip()
                if label != "right":
                    continue

                raw = detect_fist_ratio(lm,w,h)
                stable = skill.stable(raw)
                skill.update(stable)
                skill.draw(frame,lm,w,h)

        # output to GUI
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        video_label.setPixmap(QPixmap.fromImage(qimg))

        prev = time.time()

    return update

