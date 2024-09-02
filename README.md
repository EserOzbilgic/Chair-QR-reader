# chair-qr-project

Real-Time Chair Detection and QR Code Recognition with YOLOv8 and OpenCV
This project demonstrates real-time object detection and QR code recognition using the YOLOv8 model, OpenCV, and the Pyzbar library. The script captures live video from a webcam, detects chairs using YOLOv8, and identifies QR codes within the bounding boxes of detected chairs. The QR code data is then saved to a file and displayed in a separate window.

Features
Real-time Chair Detection: Utilizes the YOLOv8 model to detect chairs in the video feed.
QR Code Recognition: Detects and decodes QR codes within the chair bounding boxes using the Pyzbar library.
Data Logging: Saves the detected QR code data to a text file (Database.txt).
Multiple Display Windows:
Displays the webcam feed with chair bounding boxes and labels.
Shows detected QR code information in a separate window.

opencv-python
pyzbar
ultralytics
supervision
numpy
