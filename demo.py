import sys
import cv2
import imutils
from yoloDet import YoloTRT
import time
import numpy as np
import pyttsx3
import os
from collections import deque
from scipy.stats import linregress

# Initialize YOLO model
model = YoloTRT(library="yolov7/build/libmyplugins.so", engine="yolov7/build/best_0609.engine", conf=0.5)

# Video capture
cap = cv2.VideoCapture("videos/demo1.mov")

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

frame_count = 0
object_trajectories = {}
object_classes = {}
trajectory_length = 10
alert_ids = set()
alert_ids_close = set()
engine = pyttsx3.init()

w_threshold_percentage = 0.3
h_threshold_percentage = 0.3


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area


def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


#def distance(point1, point2):
#    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def distance(point1, point2):
    return abs(point1[0]-point2[0])


def alert_message1(obj_class):
    message = f"{obj_class} is approaching!"
    engine.say(message)
    engine.runAndWait()

def alert_message2(obj_class):
    message = f"{obj_class} is too close"
    engine.say(message)
    engine.runAndWait()

def is_approaching(trajectory, observer_position):
    if len(trajectory) < 5:
        return False
    distances = [distance(pos_size[0], observer_position) for pos_size in trajectory]
    sizes = [pos_size[1] for pos_size in trajectory]
    distance_indices = np.arange(len(distances))
    slope_distances, _, _, _, _ = linregress(distance_indices, distances)
    size_indices = np.arange(len(sizes))
    slope_sizes, _, _, _, _ = linregress(size_indices, sizes)
    return slope_distances < 0 and slope_sizes > 0

##########################################################################################
def draw_trapezoid(w, h):
    top_left = (int(w * 0.45), int(h * 0.7))
    top_right = (int(w * 0.55), int(h * 0.7))
    bottom_left = (int(w * 0.4), h)
    bottom_right = (int(w * 0.6), h)
    points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    cv2.polylines(frame, [points], isClosed=True, color=(0,255,0), thickness=2)
    return points


def is_point_in_trapezoid(point, trapezoid):
    x, y = point
    result = cv2.pointPolygonTest(trapezoid, (x, y), False)
    return result >= 0


def is_rect_in_trapezoid(box, trapezoid):
    rect_point = [(box[0], box[3]), (box[2], box[3])]
    for point in rect_point:
        if is_point_in_trapezoid(point, trapezoid):
            return True
    return False

def on_segment(p, q, r):
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def do_intersect(p1, q1, p2, q2):

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    # Special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def is_box_intersecting_trapezoid(box, trapezoid):
    """Returns True if the segment intersects any of the trapezoid's sides."""
    # Trapezoid points
    top_left, top_right, bottom_right, bottom_left = trapezoid

    # Trapezoid sides (we only consider the two non-parallel sides)
    sides = [(top_left, bottom_left), (top_right, bottom_right)]

    segment=[(box[0], box[3]), (box[2], box[3])]

    for side in sides:
        if do_intersect(side[0], side[1], segment[0], segment[1]):
            return True

    return False
############################################################################################

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of frame sequence.")
        break

    frame = imutils.resize(frame, width=600)
    original_h, original_w = frame.shape[:2]
    observer_position = (original_w / 2, original_h)
    # frame_area = original_w * original_h
    w_threshold = original_w * w_threshold_percentage
    h_threshold = original_h * h_threshold_percentage

    detections, t = model.Inference(frame)
    t3 = time.time()
    trapezoid = draw_trapezoid(original_w, original_h)
    for obj in detections:
        box = obj['box']
        # import pdb;pdb.set_trace()
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]

        if obj['type'] == 'movable':
            if w > w_threshold or h > h_threshold:
                highest_iou = 0
                matched_id = None

                for obj_id in object_trajectories:
                    if object_classes.get(obj_id) == obj['class']:
                        iou = calculate_iou([x, y, w, h], object_trajectories[obj_id][-1][2])
                        if iou > highest_iou:
                            highest_iou = iou
                            matched_id = obj_id

                if matched_id is not None and highest_iou > 0.3:
                    obj_id = matched_id
                else:
                    obj_id = len(object_trajectories)
                    object_trajectories[obj_id] = deque(maxlen=trajectory_length)

                obj['id'] = obj_id
                object_classes[obj_id] = obj['class']

                box1 = np.array([x, y, w, h], dtype=np.float32)
                object_trajectories[obj_id].append((get_center(box), w * h, box1))

                if is_rect_in_trapezoid(box, trapezoid) or is_box_intersecting_trapezoid(box, trapezoid):
                    alert_text = f"Alert! A {obj['class']} is too close!"
                    label = "Too Close {}:{:.2f} ID:{}".format(obj['class'], obj['conf'], obj_id)
                    cv2.putText(frame, alert_text, (original_w // 2 - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    model.PlotBbox(box, frame, color=(0, 0, 255), label=label)
                    cv2.polylines(frame, [trapezoid], isClosed=True, color=(0,0,255), thickness=2)
                    if obj_id not in alert_ids_close:
                        alert_ids_close.add(obj_id)
                        print("\n"*5+"="*30+ f"\n{alert_text} ID: {obj_id}\n" + "="*30+"\n"*5)
                        alert_message2(obj['class'])

                elif is_approaching(object_trajectories[obj_id], observer_position):
                    alert_text = f"Alert! A {obj['class']} is approaching!"
                    label = "Approaching {}:{:.2f} ID:{}".format(obj['class'], obj['conf'], obj_id)
                    cv2.putText(frame, alert_text, (original_w // 2 - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)
                    model.PlotBbox(box, frame, color=(0, 0, 255), label=label)

                    if obj_id not in alert_ids:
                        alert_ids.add(obj_id)
                        print("\n"*5+"="*30+ f"\n{alert_text} ID: {obj_id}\n" + "="*30+"\n"*5)
                        alert_message1(obj['class'])
                
                else:
                    label = "{}:{:.2f} ID:{}".format(obj['class'], obj['conf'], obj_id)
                    model.PlotBbox(box, frame, color=model.colors[obj['class']], label=label)
        else:
            if is_rect_in_trapezoid(box, trapezoid) or is_box_intersecting_trapezoid(box, trapezoid):
                highest_iou = 0
                matched_id = None

                for obj_id in object_trajectories:
                    if object_classes.get(obj_id) == obj['class']:
                        iou = calculate_iou([x, y, w, h], object_trajectories[obj_id][-1][2])
                        if iou > highest_iou:
                            highest_iou = iou
                            matched_id = obj_id

                if matched_id is not None and highest_iou > 0.05:
                    obj_id = matched_id
                else:
                    obj_id = len(object_trajectories)
                    object_trajectories[obj_id] = deque(maxlen=trajectory_length)

                obj['id'] = obj_id
                object_classes[obj_id] = obj['class']
                

                box1 = np.array([x, y, w, h], dtype=np.float32)
                object_trajectories[obj_id].append((get_center(box), w * h, box1))

                alert_text = f"Alert! A {obj['class']} is too close!"
                label = "Too Close {}:{:.2f} ID:{}".format(obj['class'], obj['conf'], obj_id)
                cv2.putText(frame, alert_text, (original_w // 2 - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            2)
                cv2.polylines(frame, [trapezoid], isClosed=True, color=(0,0,255), thickness=2)
                model.PlotBbox(box, frame, color=(0, 0, 255), label=label)
                if obj_id not in alert_ids_close:
                    alert_ids_close.add(obj_id)
                    print("\n"*5+"="*30+ f"\n{alert_text} ID: {obj_id}\n" + "="*30+"\n"*5)
                    alert_message2(obj['class'])
            else:
                model.PlotBbox(box, frame, color=model.colors[obj['class']], label=obj['class'])

    t4 = time.time()
    t = t + t4 - t3
    cv2.putText(frame, "FPS: {:.2f}".format(1 / t), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Output", frame)

    key = cv2.waitKey(1)
    frame_count += 1
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
