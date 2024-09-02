import cv2
from pyzbar.pyzbar import decode
import supervision as sv
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('yolov8n.pt')

# Open video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Open file for saving QR code data
with open('Database.txt', 'a') as database_file:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture image")
            break
        
        # QR code detection
        qr_codes = decode(frame)
        qr_output = ""
        
        # YOLOv8 detection
        results = model(frame)[0]
        chair_detections = results.boxes[results.boxes.cls == 56]

        if chair_detections.xyxy.shape[0] > 0:
            detections = sv.Detections(
                xyxy=chair_detections.xyxy.cpu().numpy(),
                confidence=chair_detections.conf.cpu().numpy(),
                class_id=chair_detections.cls.cpu().numpy().astype(int)
            )
            labels = ["chair"] * detections.xyxy.shape[0]
            annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            
            # Check if QR codes are within chair bounding boxes
            for qr in qr_codes:
                qr_bbox = qr.rect
                qr_center = (qr_bbox[0] + qr_bbox[2] // 2, qr_bbox[1] + qr_bbox[3] // 2)
                for det in detections.xyxy:
                    x1, y1, x2, y2 = det[:4]
                    if x1 <= qr_center[0] <= x2 and y1 <= qr_center[1] <= y2:
                        qr_data = qr.data.decode('utf-8')
                        qr_output += f"QR Code Type: {qr.type}\n"
                        qr_output += f"QR Code Data: {qr_data}\n"
                        # Save QR code data to file
                        database_file.write(f"QR Code Data: {qr_data}\n")
                        database_file.flush()  # Ensure data is written to the file immediately
        
        else:
            annotated_image = frame

        # Display the annotated image
        cv2.imshow("Webcam", annotated_image)
        
        # Display the QR code output in a separate window
        qr_output_img = 255 * np.ones((480, 640, 3), dtype=np.uint8)  # White image
        y0, dy = 30, 30
        for i, line in enumerate(qr_output.split('\n')):
            y = y0 + i * dy
            cv2.putText(qr_output_img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("QR Code Output", qr_output_img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
