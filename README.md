Observatory Guardian AI - Hybrid Allsky Safety Monitor

Observatory Guardian AI is a sophisticated, hybrid safety monitoring system designed for astronomical observatories. It combines Local AI image classification (TensorFlow Lite) with real-time weather station telemetry (Ecowitt) to provide a high-reliability "Safe/Unsafe" signal for ASCOM and Alpaca-compatible imaging software.

🌟 Key Features

AI Sky Classification: Uses a .tflite model to analyze Allsky images in real-time, detecting conditions like Clear, Cloudy, or Rain.

Modular Operation: Works as a standalone AI monitor using only your Allsky camera image, or as a hybrid system when paired with a weather station.

Weather Station Integration: Optional native HTTP listener for Ecowitt weather stations to monitor wind speed, rain rate, humidity, and solar radiation.

ASCOM Alpaca Ready: Implements a full Alpaca SafetyMonitor API, allowing software like N.I.N.A., SGP, or Voyager to connect over the network.

Hybrid Safety Logic: * Immediate Trigger: Hard-wired "Unsafe" flip for rain or high wind.

Cloud Grace Period: User-definable countdown timer for transient clouds to prevent unnecessary observatory shutdowns.

Web Dashboard: A built-in responsive web server providing live charts (Chart.js), compass directions, and AI confidence metrics.

Data Proxying: Acts as a bridge to forward Ecowitt telemetry to Home Assistant or other third-party services.

Flexible Image Ingestion: Supports MQTT (fastest), SFTP (Allsky Map compatible), or IP Camera (RTSP/MJPEG) streams.

🏗️ System Architecture

Image Source: Allsky camera uploads images via MQTT or SFTP.

AI Engine: The Python script crops the image and runs it through a TensorFlow Lite classifier.

Weather Listener (Optional): Ecowitt station sends "Custom Server" packets to the script on a dedicated port.

Decision Logic: The system evaluates available inputs. If the weather station is disabled, safety is determined solely by AI image analysis.

Output: * Updates a local ASCOM_STATUS.txt file.

Serves the Alpaca /api/v1/safetymonitor/ endpoint.

Forwards weather data to Home Assistant (if configured).

🚀 Installation

Prerequisites

Python 3.8 or higher.

A trained TensorFlow Lite model (.tflite) and corresponding labels.txt.

Dependencies

pip install opencv-python numpy tensorflow paramiko paho-mqtt Pillow requests


Running the Monitor

Clone this repository.

Run the script:

python "Allsky safety Moniter-AI.py"


Use the ⚙ Configure Settings button in the GUI to set your paths and IP addresses.

⚙️ Configuration Guide

1. AI-Only vs. Hybrid Mode

The system is designed to be flexible:

AI Only: Disable "Use Weather Station" in settings. Safety will be based on the AI's classification of the sky image.

Hybrid: Enable "Use Weather Station". The system will mark the state as UNSAFE if either the AI detects rain/clouds or the weather station reports high winds/rain.

2. AI Logic

Model Path: Select your .tflite file.

Safe AI Labels: A comma-separated list of labels that represent "Safe" conditions (e.g., Clear, Partially Clear).

Cloud Grace: Number of minutes the AI will wait while "Cloudy" before marking the session as Unsafe.

3. Weather Station (Ecowitt)

In your Ecowitt App/WS View, set the Custom Server to:

Protocol: HTTP / POST

Server IP: IP of the computer running this script.

Port: Default is 8080.

Interval: 16-60 seconds.

4. Home Assistant Integration (Proxy)

Enable Forwarding in the Weather tab.

Input your Home Assistant Ecowitt Webhook URL.

📊 Safety Trigger Matrix

Source

Trigger

State

Immediate?

Ecowitt (Optional)

Rain Rate > 0

UNSAFE

✅ YES

Ecowitt (Optional)

Wind > Limit

UNSAFE

✅ YES

AI Engine

"Rain" Label

UNSAFE

✅ YES

AI Engine

"Cloudy" Label

UNSAFE

⏳ After Grace Period

System

Stale Image

UNSAFE

✅ YES

⚖️ License & Disclaimer

This software is provided "as is". Building an automated observatory involves risks to expensive equipment. Always ensure you have physical fail-safes (like a local rain sensor) in addition to this software.

Developed for the Amateur Astronomy Community.

🧠 Phase 3: Training Your AI Model (Replication Guide)

If you need to retrain the model with more data (e.g., adding a Fog class) or adapt it to a new camera, follow these steps:

Prepare Data:

Gather raw images and run the preprocessing script (allsky_image_prep.py) to create 224x224 images that are center-cropped ((1300x1300)example for my camera it can be changed in the script for your suituation) .

Organize the pre-processed images into folders named by their class (e.g., training_data/Cloudy).

Compress the entire folder structure into a single training_data.zip file.

Train in Colab:

Upload the training_data.zip file to a Google Colab notebook.

Run the provided colab_AI_training_script.py (The code handles loading MobileNetV2, training the classification head, and exporting).

Export TFLite:

The script automatically generates two files:

allsky_cloud_detector_final.tflite (The new model)

labels.txt (The updated list of classes)

Download these two files and use them to replace the old files in your observatory deployment folder. The AllSkyMonitor.exe will automatically load the improved model upon restart.
