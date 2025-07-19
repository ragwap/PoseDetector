# PoseDetector

Just run the following commands one after another once you clone the repo inside the project root. <br />
<i>**Note**: Omit the suffix '3' in python and pip commands if you are using windows or mac </i>
```
python3 -m venv venv
```
```
source ./venv/bin/activate
```
Command to activate the python virtual env might depend on the terminal/kernel you are using. These commands are for bash users. If you are using a different terminal, please change accordingly
```
pip3 install -r requirements.txt
```
```
python3 pose_detector.py
```

## Results

As shown in the screenshots, videos with a large number of people can appear visually congested due to the overlapping detected poses and keypoints. However, when only a few individuals are present, the head directions and poses are displayed much more clearly. <br />

<img title="1" src="Screenshot 2025-07-20 000407.png">
<img title="2" src="Screenshot 2025-07-20 000524.png">
<img title="3" src="Screenshot 2025-07-20 000752.png">
<img title="4" src="Screenshot 2025-07-20 001048.png">
<img title="5" src="Screenshot 2025-07-20 001556.png">
