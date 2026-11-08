# cache-only-challengeX
Repository of the cache only team for the fall 2026 ChallengeX event. Our demonstration uses

## The vision model
This repository, due to the size of the model, does not contain our vision model used in this project. To see the most recent dataset used for this project, see https://app.roboflow.com/jam-workspace/challenge-x-turtle-dataset/models.

# How to use
## Vision Model
Since the model does not exist in the repository currently, it must be trained either via this code or by another source such as roboflow. Our initial set of images used for the vision model are included in this repository under the train/ test/ and valid/ folders. If you wish to train the model off of this dataset, run `python train.py`. Be noted that this training process takes a long time. You may uncomment the device="cuda" line if your device has CUDA available, though this may not be the case. CUDA requirements can be installed through https://pytorch.org alongside an appropriate Nvidia GPU.

## Video Detection
For saved video footage, the only file that needs to be used is "receive_gps.py". Make sure that whatever video you want to test detection on is saved in the proper folder (by default test_videos/) and that the "TEST_VIDEO" constant is set to the video you would like. A few sample videos are included in this repository. Make sure in "def main():" that it calls "capture_video()" as the live feed and video functions live in the same file and are run through the main method. Once this is checked, simply run `python make_detections.py`

## Live Feed
The live feed requires the pyrtmp library. You can test to ensure the library is functioning properly by following the guide at https://github.com/Eittipat/pyrtmp. For live feed to work, both the drone controller and whatever device is running this code must be on the same network. Ensure the drone has RTMP capabilities and is streaming to the IPV4 of your device running python and the port it's listening to as well as a location for the stream to save if desired (i.e. rtmp://192.168.1.20:1935/live/drone). When this is established, ensure "def main():" calls "capture_live()" then run `python rtmp_server.py` in one terminal and `python make_detections.py` in another. Once the rtmp_server is listening, you can start the drone live feed and the code should run. Detections will be saved by default in the detections/ library.

Be warned that against sufficiently powerful devices, the live video feed may come back as laggy or unresponsive, in which case you can disable seeing the live feed by removing the display window.

## GPS
**The GPS files are experiemental and have not been sufficiently tested.** This code was tested on two separate meshtastic nodes (T1000-E and Tracker L1 Pro). Ensure both devices use the same communication channel and frequency and that they can appropriately communicate. Connect one node to the device running the python code and run `python gps_test.py` to test to see if communications being seen. If this works, `python receive_gps.py` should work and send periodic gps updates.

If using meshtastic nodes, certain drivers may be required, see https://meshtastic.org/docs/development/python/library/ for information.

There may also be a way to view GPS coordinates from a drone's "Remote ID (RID)" which drones over 250g must legally transmit during flight. This has been researched but not implemented, there may be existing libraries which can do this, but due to time constraints they were not tested.