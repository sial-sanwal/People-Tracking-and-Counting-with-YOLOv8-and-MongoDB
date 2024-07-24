import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import *
import os
import datetime
from pymongo import MongoClient
import time
import matplotlib.pyplot as plt

# Function to handle mouse events for drawing a custom line
def draw_custom_line(event, x, y, flags, param):
    global line_start, line_end, line_drawn

    if not line_drawn:
        if event == cv2.EVENT_LBUTTONDOWN:
            line_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            line_end = (x, y)
            line_drawn = True

# Initialize YOLO model and other variables
model = YOLO('./model_weights/yolov8s.pt')
line_start = None
line_end = None
line_drawn = False
tracker = Tracker()
offset = 6
up_counter = 0
down_counter = 0
person_positions = {}
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', draw_custom_line)

# Create a separate window for "Statistics"
cv2.namedWindow('Statistics', cv2.WINDOW_NORMAL)
cv2.moveWindow('Statistics', 50, 50)

# Open the video capture
cap = cv2.VideoCapture('videoclip.mp4')

# Load class names for YOLO
with open("coco.txt", "r") as my_file:
    data = my_file.read()
    class_list = data.split("\n")

# Connect to MongoDB
connection_string = "YOUR CONNECTION STRING"
client = MongoClient(connection_string)
db = client.get_database("DATABASE NAME")
collection = db["COLLECTION NSME"]

# Initialize the start_time variable as datetime.datetime object
start_time = datetime.datetime.now()  # Convert start_time to datetime object

# Lists to store data for the graph
time_points = []
in_counts = []
out_counts = []

# Main loop for processing frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))

    if not line_drawn:
        cv2.imshow("RGB", frame)
    else:
        # Object detection and tracking
        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        list = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            c = class_list[d]
            if 'person' in c:
                list.append([x1, y1, x2, y2])

        bbox_id = tracker.update(list)
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            if id not in person_positions:
                person_positions[id] = []

            person_positions[id].append(cy)

            # Draw bounding box, centroid, and ID
            if line_start and line_end:
                cy1 = min(line_start[1], line_end[1])
                cy2 = max(line_start[1], line_end[1])

                if cy < (cy1 + cy2) // 2 - offset:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Red
                elif cy > (cy1 + cy2) // 2 + offset:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Green

                cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)
                cv2.putText(frame, f'ID {id}', (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw lines and count persons crossing
        cv2.line(frame, (line_start[0], line_start[1]), (line_end[0], line_end[1]), (0, 255, 0), 2)
        cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
        cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

        for id, positions in person_positions.items():
            if len(positions) >= 2:
                if positions[-2] <= cy1 and positions[-1] > cy1:
                    down_counter += 1
                elif positions[-2] >= cy2 and positions[-1] < cy2:
                    up_counter += 1

        # Update and display counters in the "RGB" window
        counter_text = f'UP: {up_counter} | DOWN: {down_counter}'
        cv2.putText(frame, counter_text, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Insert data into MongoDB
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.datetime.now()
        record_interval = 5  # Define the record interval in seconds
        if (current_time - start_time).total_seconds() >= record_interval:
            # Store the updated person_positions data in MongoDB
            if person_positions:
                collection.insert_one({
                    'date': current_date,
                    'timestamp': current_time,
                    'up_counter': up_counter,
                    'down_counter': down_counter
                })

            start_time = current_time

        # Display the frame
        cv2.imshow("RGB", frame)

        # Update data for the graph
        time_points.append(current_time.strftime("%H:%M:%S"))
        in_counts.append(up_counter)
        out_counts.append(down_counter)

        # Plot the graph with counters
        plt.figure(figsize=(10, 6))
        plt.plot(time_points, in_counts, label='In', color='blue')
        plt.plot(time_points, out_counts, label='Out', color='green')
        plt.xlabel('Time')
        plt.ylabel('Counts')
        plt.title('People Counts Over Time')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        # Convert the plot to an image and display it in the "Statistics" window
        graph_img = plt.gcf()
        graph_img.canvas.draw()
        graph_data = np.frombuffer(graph_img.canvas.tostring_rgb(), dtype=np.uint8)
        graph_data = graph_data.reshape(graph_img.canvas.get_width_height()[::-1] + (3,))

        # Create a frame for the "Statistics" window
        statistics_frame = np.zeros((graph_data.shape[0] + 100, graph_data.shape[1], 3), dtype=np.uint8)

        # Add the graph image to the "Statistics" frame
        statistics_frame[0:graph_data.shape[0], :] = graph_data

        # Add "in" and "out" counters to the "Statistics" frame
        counters_text = f'IN: {up_counter} | OUT: {down_counter}'
        cv2.putText(statistics_frame, counters_text, (10, graph_data.shape[0] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the "Statistics" frame
        cv2.imshow('Statistics', statistics_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
