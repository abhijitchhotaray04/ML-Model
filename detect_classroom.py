import cv2
from ultralytics import YOLO

# 1. Load pre-trained YOLO11n (~5.5 MB auto-downloads once on first run)
model = YOLO("yolo11n.pt")  #

# 2. Exact COCO class IDs for your classroom items
CLASSROOM_CLASSES = [
    0,   # person (human)
    13,  # bench
    24,  # backpack (bag)
    26,  # handbag (bag)
    39,  # bottle
    56,  # chair
    62,  # tv
    63,  # laptop
    67,  # cell phone
    73,  # book
]

# 3. Open your laptop's primary webcam (0 = built-in camera)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions or index.")

print("Classroom Detector running... Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection filtered ONLY to classroom classes on CPU
    results = model.predict(
        source=frame,
        classes=CLASSROOM_CLASSES,
        conf=0.40,
        device="cpu",
        verbose=False
    )

    # Count detected objects per category in the current frame
    counts = {}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        counts[label] = counts.get(label, 0) + 1

    # Draw bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Overlay live object counts in the top-left corner
    y_offset = 30
    cv2.putText(annotated_frame, "Classroom Live Count:", (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for label, count in sorted(counts.items()):
        y_offset += 28
        cv2.putText(annotated_frame, f"{label}: {count}", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Show the live feed
    cv2.imshow("Classroom Object Detector (YOLO11n)", annotated_frame)

    # Exit cleanly when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()