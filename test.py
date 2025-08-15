import logging
from collections import deque
from ultralytics import YOLO
import cv2
import numpy as np

# Suppress logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load both models
det_model = YOLO("yolov8s.pt")            # for sports ball detection/tracking
pose_model = YOLO("yolov8s-pose.pt")      # for pose estimation (person only)

# Run tracking on ball
results = det_model.track(
    source="videos/test.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    stream=True,
    save=False
)

out = None
N = 5
movement_threshold = 2.0
centroid_history = {}

# Pose skeleton pairs (COCO)
POSE_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Main loop
for result in results:
    frame = result.orig_img.copy()

    if out is None:
        h, w = frame.shape[:2]
        out = cv2.VideoWriter(
            'outputs/annotated_test_2.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,
            (w, h)
        )

    # ------------------ POSE ESTIMATION ------------------
    pose_result = pose_model.predict(frame, stream=False, verbose=False)[0]

    if hasattr(pose_result, "boxes") and hasattr(pose_result, "keypoints"):
        for box, kps in zip(pose_result.boxes, pose_result.keypoints.data):
            cls_id = int(box.cls[0])
            if pose_model.names[cls_id] != "person":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            kps = kps.cpu().numpy().reshape(-1, 3)

            for x, y, conf in kps:
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            for i, j in POSE_SKELETON:
                if kps[i][2] > 0.3 and kps[j][2] > 0.3:
                    pt1 = (int(kps[i][0]), int(kps[i][1]))
                    pt2 = (int(kps[j][0]), int(kps[j][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

    # ------------------ BALL TRACKING + STATE ------------------
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if det_model.names[cls_id] != "sports ball":
            continue

        if box.id is None:
            continue
        obj_id = int(box.id.item())

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        conf = box.conf.item()
        area = (x2 - x1) * (y2 - y1)

        if conf < 0.4 or area > 8000:
            continue

        if obj_id not in centroid_history:
            centroid_history[obj_id] = deque(maxlen=N)
        centroid_history[obj_id].append((cx, cy))

        state = "STATIONARY"
        pts = centroid_history[obj_id]
        if len(pts) >= 2:
            dists = [
                np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                for i in range(1, len(pts))
            ]
            avg_move = sum(dists) / len(dists)
            if avg_move > movement_threshold:
                state = "ACTION"
        else:
            avg_move = 0.0

        label = f"ID {obj_id} | {state} | V={avg_move:.1f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2
        )

    out.write(frame)
    cv2.imshow("Tracking + Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cv2.destroyAllWindows()
