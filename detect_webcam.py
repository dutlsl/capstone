import cv2, time, collections
from ultralytics import YOLO

# ──────────── 사용자 설정 ────────────
MODEL     = "yolov5n.pt"
CAM_IDX   = 0
CONF      = 0.35
FPS_EST   = 30              # 카메라 실제 FPS
MIN_AREA  = 8000            # 너무 작은 박스 무시
CENTER_ROI = 0.4            # 화면 중앙 40% 만 사용
V_REF     = 250             # px/s (캠 해상도 640×480 기준)
WIN_SIZE  = 8               # 최근 0.5 s (8프레임) 이동 평균
UP_REQUIRE = 3              # 위험 승급엔 연속 3프레임 필요
DOWN_DELAY = 30             # 1 s 연속 안전 시 하강
TRIGGER_LVL = 7             # 진동 시작 위험도
# ────────────────────────────────────

model = YOLO(MODEL)
cap   = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

hist = collections.deque(maxlen=WIN_SIZE)           # (t, area, cy)
consecutive_approach = 0
consecutive_safe     = 0
risk_level           = 1

def danger_mapping(s_norm):
    # 간단 단계표 : 높은 속도에만 높은 레벨
    if s_norm < 0.20: return 3
    if s_norm < 0.35: return 5
    if s_norm < 0.55: return 6
    if s_norm < 0.75: return 7
    if s_norm < 0.90: return 8
    return 9

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t_now = time.time()

    results = model(frame, conf=CONF)
    h, w, _ = frame.shape
    x_min, x_max = int(w*(0.5-CENTER_ROI/2)), int(w*(0.5+CENTER_ROI/2))

    target = None
    # 가장 큰 차량·오토바이·자전거 중 중앙 영역에 있는 것 1개
    for box in results[0].boxes:
        cls = results[0].names[int(box.cls[0])]
        if cls not in ('car', 'motorcycle', 'bicycle'):
            continue
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        if x2 < x_min or x1 > x_max:
            continue                              # ROI 밖
        area = (x2-x1)*(y2-y1)
        if area < MIN_AREA:
            continue
        if target is None or area > target['area']:
            target = {'cls':cls, 'area':area, 'cx':(x1+x2)/2, 'cy':(y1+y2)/2}

    if target:
        area, cy = target['area'], target['cy']
        # 이동 평균 계산
        if hist:
            dt   = t_now - hist[-1][0]
            vy   = (cy - hist[-1][2]) / max(dt, 1e-3)
            ratio = area / hist[-1][1]
        else:
            vy, ratio = 0, 1
        hist.append((t_now, area, cy))

        # 최근 WIN_SIZE 프레임 평균 속도 px/s
        if len(hist) >= 2:
            vy_avg = abs((hist[-1][2] - hist[0][2]) / (hist[-1][0] - hist[0][0]))
        else:
            vy_avg = 0
        s_norm = min(1.0, vy_avg / V_REF)

        # 접근 조건(둘 다 만족)
        approaching = (ratio > 1.10) and (vy < -15)

        if approaching:
            consecutive_safe = 0
            consecutive_approach += 1
            if consecutive_approach >= UP_REQUIRE and risk_level < 10:
                risk_level = max(risk_level, danger_mapping(s_norm))
        else:
            consecutive_approach = 0
            consecutive_safe += 1
            if consecutive_safe >= DOWN_DELAY and risk_level > 1:
                risk_level -= 1
                consecutive_safe = 0

        print(f"{target['cls']} {risk_level}")
        # 진동 모터 연동 시: if risk_level >= TRIGGER_LVL: pwm_on()
    else:
        hist.clear()
        consecutive_approach = 0
        consecutive_safe += 1
        if consecutive_safe >= DOWN_DELAY and risk_level > 1:
            risk_level -= 1
            consecutive_safe = 0
        print("none", risk_level)

    cv2.imshow("Detection", results[0].plot())
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
