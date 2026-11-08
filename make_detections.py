# RD-DETR documents can be found here: https://rfdetr.roboflow.com/latest/

import threading
import time
import os
import supervision as sv
import cv2
from rfdetr import RFDETRSmall
import json
from datetime import datetime
import base64

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay"
model = RFDETRSmall(pretrain_weights="runs/drone_v1/checkpoint_best_total.pth")

DETECTION_THRESHOLD = 0.3       # Confidence of detection required for the detection to be displayed
FRAMES_TO_SAVE = 30             # Number of simultaneous detection frames until image is saved
DETECTION_LENIENCY = 10         # Number of frames a detection can drop before resetting detection count
SAVE_DELAY = 60                 # Number of frames with no detection before another image can save
DETECTION_DIR = "detections"    # Directory name for detections to be saved to
TEST_VIDEO = "test_videos/turtle_burrow.mp4"

# Thickness and scale of the detection box and confidence label
box_annotator = sv.BoxAnnotator(
    thickness=3
)
label_annotator = sv.LabelAnnotator(
    text_scale=1.5,
    text_thickness=3
)

"""
    Class for detecting each frame in a live video feed
"""
class FrameReader:
    def __init__(self, source):
        self.source = source
        self.frame = None
        self.frame_id = 0
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
 
    def _read_loop(self):
        while self.running:
            video_capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            while self.running and video_capture.isOpened():
                success, frame_bgr = video_capture.read()
                if not success:
                    break
                self.frame = frame_bgr
                self.frame_id += 1
            video_capture.release()
            time.sleep(1)  # stream not started yet, or it dropped; try again
 
    def stop(self):
        self.running = False

"""
    Function for displaying detections and confidence of videos saved on the system.
    Change constant value "TEST_VIDEO" at the top of the file to whichever input file you wish to test.
    This function does not save output result, it is instead used to test if the vision model is working
    before putting it to live test.
"""
def capture_video():
    video_source = TEST_VIDEO
    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")

    cv2.namedWindow("RF-DETR Video", cv2.WINDOW_NORMAL) # initialize window
    cv2.resizeWindow("RF-DETR Video", 1024, 1024)       # resize window
    print("Window opening, press 'q' to exit window")

    while True:
        success, frame_bgr = video_capture.read()
        if not success:
            break


        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB format
        detections = model.predict(frame_rgb, threshold=DETECTION_THRESHOLD)

        labels = [
            f"{model.class_names[class_id]} {confidence:.2%}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = box_annotator.annotate(frame_bgr, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

        cv2.imshow("RF-DETR Video", annotated_frame)    # display window

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def capture_live():
    reader = FrameReader("udp://127.0.0.1:5000?fifo_size=5000000&overrun_nonfatal=1")
    last_frame_id = 0
    filenum = 1
    detect_streak = 0
    miss_streak = 0
    detection_saved = False

    cv2.namedWindow("RF-DETR Video", cv2.WINDOW_NORMAL) # initialize window
    cv2.resizeWindow("RF-DETR Video", 512, 512)         # resize window
    print("Window opening, press 'q' to exit window")

    while True:
            # nothing new to process yet: keep the window responsive and wait
            if reader.frame is None or reader.frame_id == last_frame_id:
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
                continue

            last_frame_id = reader.frame_id
            frame_bgr = reader.frame.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # convert to RGB format
            detections = model.predict(frame_rgb, threshold=DETECTION_THRESHOLD)
            detections = detections[detections.class_id < len(model.class_names)]   # remove objects that aren't real detections

            labels = [
                        f"{model.class_names[class_id]} {confidence:.2%}"
                        for class_id, confidence in zip(detections.class_id, detections.confidence)
                    ]
            
            annotated_frame = box_annotator.annotate(frame_bgr, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

            if len(detections) > 0:
                detect_streak += 1
                miss_streak = 0

            # If an object is detected for enough continuous frames, save the image and proper json data
            if detect_streak >= FRAMES_TO_SAVE and detection_saved == False:
                event = save_detection(annotated_frame, detections, filenum)
                print(f"Saved {event['image_file']} with confidence {event['max_confidence']}")
                filenum += 1
                detection_saved = True

            # If the live feed hasn't detected an object for a long time, it's assumed that the previous
            # detection has been passed and new detections are new objects
            elif miss_streak > SAVE_DELAY:
                detection_saved = False

            # If no object is detected, increment a miss streak and reset the detection streak if said
            # miss streak is past a certain leniency threshold
            else:
                miss_streak += 1
                if (miss_streak % 5 == 0):
                    print(f"Miss streak: {miss_streak}")
                if miss_streak > DETECTION_LENIENCY:
                    detect_streak = 0

            cv2.imshow("RF-DETR Video", annotated_frame)        # display window

    # Might want to clear out detections directory here during actual implementation
    reader.stop()
    cv2.destroyAllWindows()


"""
    Function for saving the image and json data of a recent detection to specified directory. 
    The directory can be changed by changing the constant value "DETECTION_DIR" at the top
    of the file.
"""
def save_detection(annotated_frame, detections, file_number):
    os.makedirs(DETECTION_DIR, exist_ok=True)
    filenum = file_number

    while os.path.exists(f"{DETECTION_DIR}/detection_{filenum}.jpg"):
        filenum += 1

    cv2.imwrite(f"{DETECTION_DIR}/detection_{filenum}.jpg", annotated_frame)
    success, encoded_image = cv2.imencode('.jpg', annotated_frame)

    if not success:
        print(f"Failed to encode image: {DETECTION_DIR}/detection_{filenum}.jpg")

    # Json data to be saved to the output file
    event = {
        "id": filenum,
        "time": datetime.now().isoformat(),
        "classes": sorted({model.class_names[c] for c in detections.class_id}),
        "max_confidence": round(float(detections.confidence.max()), 3),
        "num_detections": len(detections),
        "image_file": f"{DETECTION_DIR}/detection_{filenum}",
        "image_b64": base64.b64encode(encoded_image.tobytes()).decode("ascii"),
        "gps": None,   # add later
    }
    with open(f"{DETECTION_DIR}/detection_{filenum}.json", "w") as f:
        json.dump(event, f, indent=4)
    return event

def main():
    capture_video()

if __name__ == "__main__":
    main()
