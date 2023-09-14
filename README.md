# Sobriety Test
The goal of this project is to address this issue by developing a new method for detecting intoxication. Our method employs the accelerometer and gyroscope built into the user's phone to perform the Tandem Gait walking test. In this test, the individual takes five heel-to-toe steps, then turns 180° and repeats the process.
The phone's data is then used to classify the user's level of intoxication. We believe our method has the potential to be a useful tool in the prevention of intoxication-related injuries.

**How to run this project**
1) Download or clone this repository (use % git clone https://github.com/CS328-Spring2023/project-proposal-group_13.git)
2) With the data provided in *src/data*, run *activity-classification-train.py* to train the model and store a .pickle file with is used as the classifier
3) If you want you can add more data using the Sensor Logger app, convert to CSV format and export in .zip format
4) Connect your laptop to your phone hotspot. 
5) Reopen the app and navigate to the “Logger” tab.
6) Tap on the radio button beside the word “Accelerometer” and "Gyroscope".
7) Run *sensor_logger.py* and go to the url which shows on the terminal
8) Press record on your app and perform the test
9) The result will show on the server
