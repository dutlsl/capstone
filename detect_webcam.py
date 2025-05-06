import cv2, time, collections
from ultralytics import YOLO

# ────────────────────────── 설정값 ──────────────────────────
MODEL_PATH   = "yolov5n.pt"   # 학습된 가중치
DEVICE_IDX   = 0              # 카메라 인덱스
CONF_THRES   = 0.30
HIST_LEN     = 10             # 최근 N프레임 저장
V_REF_PIXELS = 200            # '매우 빠른 접근' 기준 속도(px/s)
# ───────────────────────────────────────────────────────────

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# (timestamp, area, cy) 히스토리
history = collections.deque(maxlen=HIST_LEN)

def norm_speed(area_now, cy_now, t_now):
    """정규화 속도 s(0~1)와 접근 여부 반환"""
    if not history:
        return 0.0, False
    t_prev, area_prev, cy_prev = history[-1]
    dt = t_now - t_prev
    if dt == 0 or area_prev == 0:
        return 0.0, False

    ratio = area_now / area_prev
    vy    = (cy_now - cy_prev) / dt       # 음수이면 카메라 방향

    approach = (ratio > 1.02) or (vy < -5)
    s        = min(1.0, abs(vy) / V_REF_PIXELS)
    return s, approach

def risk_score(s, approach):
    """정규화 속도·접근 여부 → 위험도 1~10"""
    if not approach:
        return 1 if s < 0.10 else 2 if s < 0.25 else 3
    if s < 0.10:  return 1
    if s < 0.25:  return 3
    if s < 0.40:  return 4
    if s < 0.55:  return 5
    if s < 0.70:  return 6
    if s < 0.80:  return 7
    if s < 0.90:  return 8
    if s < 0.97:  return 9
    return 10

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        t_now  = time.time()
        results = model(frame, conf=CONF_THRES)
        annotated = results[0].plot()

        # 관심 객체 중 가장 큰 박스 1개만 평가
        target_area, target_box, target_cls = 0, None, ""
        for box in results[0].boxes:
            cls_name = results[0].names[int(box.cls[0])]
            if cls_name not in ('car', 'motorcycle', 'bicycle'):
                continue
            x1,y1,x2,y2 = box.xyxy[0]
            area = (x2-x1)*(y2-y1)
            if area > target_area:
                target_area, target_box, target_cls = area, (x1,y1,x2,y2), cls_name

        if target_box is not None:
            x1,y1,x2,y2 = target_box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            s, approach = norm_speed(target_area, cy, t_now)
            score = risk_score(s, approach)
            print(f"{target_cls} {score}")

            history.append((t_now, target_area, cy))
        else:
            history.clear()            # 대상 없으면 히스토리 초기화
            print("none 1")

        cv2.imshow("YOLO Detection", annotated)
        if cv2.waitKey(1) & 0xFF == 27:    # ESC 키
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
