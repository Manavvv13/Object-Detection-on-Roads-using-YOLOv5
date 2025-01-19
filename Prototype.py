import streamlit as st
import cv2
import torch
import pycocotools
from pycocotools.coco import COCO
import numpy as np
from tempfile import NamedTemporaryFile

# Load COCO annotations
annotation_file = r'C:\Users\HP\Documents\ObjectDetection\instances_val2017.json'
coco = COCO(annotation_file)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Mapping of YOLO class IDs to COCO category IDs
yolo_to_coco_map = {
    0: 1,    # 'person' in YOLO -> 'person' in COCO
    1: 2,    # 'bicycle' -> 'bicycle'
    2: 3,    # 'car' -> 'car'
    3: 4,    # 'motorbike' -> 'motorbike'
    5: 6,    # 'bus' -> 'bus'
    7: 8,    # 'truck' -> 'truck'
    9: 10,   # 'traffic light' -> 'traffic light'
    10: 11,  # 'fire hydrant' -> 'fire hydrant'
    11: 12,  # 'stop sign' -> 'stop sign'
    # You can add more categories based on your needs
}

# List of YOLO class IDs that are relevant for road objects
relevant_yolo_classes = [2, 3, 5, 7, 9, 10, 0, 1, 11]  # car, motorbike, bus, truck, traffic light, stop sign, person

# Detection function
def detect_objects_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    # Set the window size
    window_name = "YOLO Object Detection in Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.resizeWindow(window_name, 800, 600)  # Resize window to 800x600 or any preferred size

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference with YOLOv5 model
        results = model(frame)
        results.render()  # This modifies the frame with bounding boxes and labels

        # Loop over detected objects and add category names
        for *box, conf, cls in results.xywh[0]:  # iterate over detections
            category_id = int(cls.item())  # YOLO class ID

            # Check if the detected object is in the relevant categories
            if category_id in relevant_yolo_classes:
                # Map YOLO class ID to COCO category ID
                coco_category_id = yolo_to_coco_map.get(category_id, -1)

                if coco_category_id != -1:
                    # Get category name from COCO annotations
                    category_name = coco.loadCats(coco_category_id)[0]['name']
                else:
                    category_name = "Unknown Category"  # Default if not mapped

                # Add category label to the frame
                cv2.putText(frame, f'{category_name} {conf:.2f}', 
                            (int(box[0]), int(box[1])), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with detection annotations in the resized window
        cv2.imshow(window_name, frame)

        # Break the loop on 't' key press
        if cv2.waitKey(1) & 0xFF == ord('t'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit file uploader for video
def main():
    st.title("YOLO Object Detection on Video")
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file as a temporary file
        with NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmpfile_path = tmpfile.name

        # Call the object detection function
        detect_objects_from_video(tmpfile_path)

if __name__ == "__main__":
    main()
