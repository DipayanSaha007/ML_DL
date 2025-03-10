from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (pretrained on COCO dataset)
model = YOLO('yolov8n.pt')  # Replace 'yolov8n.pt' with other weights like yolov8s.pt, yolov8m.pt, etc.

# Load an image
image_path = r"C:\Users\User\OneDrive\Pictures\street-people-cars-urban-scene-cityscape-skyscrapers-new-yor.webp"
image = cv2.imread(image_path)

# Perform inference
results = model(image)

# Display results
for result in results:
    boxes = result.boxes  # Detected bounding boxes
    for box in boxes:
        # Get box details
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class ID
        label = model.names[class_id]  # Class name

        # Draw the bounding box and label on the image
        color = (0, 255, 0)  # Green for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show the output
cv2.imshow("YOLOv8 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()