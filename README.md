# Allsky-camera-based-safety-Moniter
A robust, custom-trained, AI-powered observatory safety monitor designed to analyze real-time images from an All-Sky Camera (e.g., INDI/Raspberry Pi) and provide an immediate, reliable status to ASCOM automation software (like N.I.N.A., SGP, Voyager).

This system addresses common observatory challenges by using a highly compatible TensorFlow Lite (TFLite) model and an SFTP transfer method within a simple, multi-threaded Windows application.


![clear](https://github.com/user-attachments/assets/13ccb126-9997-4e8d-89c5-9868913d591c)

✨ Features

Custom AI Classification: Uses a custom-trained TFLite model (MobileNetV2 architecture) for superior cloud/sky condition detection compared to generic sensors.

Live Image Fetch: Securely pulls the latest image from the remote All-Sky Camera (Raspberry Pi) via SFTP with built-in retry logic.

Persistent GUI Application: A standalone Windows executable (.exe) with a user interface, running the monitoring loop in a background thread for non-blocking operation.

Configurable Safety: Safety status (IsSafe=True/False) is written to a dedicated text file, read by the ASCOM Generic File Safety Monitor.

Zero Dependencies: The final application is bundled with all necessary libraries (TensorFlow, OpenCV, Paramiko) using PyInstaller.

🚀 Phase 1: Deployment & Installation

The final application is compiled into a single executable file.

Prerequisites (Run on Observatory PC)

Your observatory PC needs only one non-Python prerequisite:

Microsoft Visual C++ Redistributable (x64): Required for TensorFlow's core components to run. If you encounter an msvcp140_1.dll error, download and install the latest x64 version from the Microsoft website.

Installation

Create Folder: Create a dedicated, non-protected folder on your PC (e.g., C:\AllskyMonitor).

Copy Files: Copy the following four files into that folder:

AllSkyMonitor.exe (The compiled application)

allsky_cloud_detector_final.tflite (Your AI Model)

labels.txt (Your list of classes)

allsky_monitor_config.json (The Configuration file)

Run Application: Double-click AllSkyMonitor.exe. (https://drive.google.com/file/d/1WtXlokWeKHyJTJKy80lLpFs5Q6R-QhhN/view?usp=sharing)

⚙️ Phase 2: Configuration and Setup

![specs](https://github.com/user-attachments/assets/417eebc6-7f50-47de-81ac-e59cec102dcc)

The application will launch, but you must configure the paths and SFTP credentials via the Settings panel before the AI model can run.

Open Settings: Click the ⚙️ Settings button in the main application window.

Configure Paths & Credentials: Fill in all fields in the Settings window:

ASCOM Status File Path: Crucial! Use the Browse button to select a file path in a non-protected folder (e.g., C:\Users\YourUser\Documents\ASCOM_STATUS.txt). This is the file your ASCOM driver will read.

AI Model Paths: Point these to the .tflite and .txt files in your installation directory.

SFTP Connection: Enter the IP address, username, password, and the full remote path to your latest.jpg file on the Allsky Camera (e.g., /home/pi/allsky/images/latest.jpg).

Safe Conditions: Enter a comma-separated list of every class name that you consider safe for observing (e.g., Clear, Partially Clear, Clear with Moon).

Save & Restart: Click Save & Restart. The application will now start the monitoring thread, pull the image, and begin updating the ASCOM file.

🔗 Phase 3: ASCOM Integration

This is the final step to link the AI output to your dome controller.

Select Driver: In your primary automation software (e.g., N.I.N.A.), select the Generic File Safety Monitor as your Safety Monitor device.

Open Properties: Click the Properties button to open the configuration window.

Set File Path: Set the File to monitor field to the exact path you chose in the application settings (e.g., C:\Users\YourUser\Documents\ASCOM_STATUS.txt).

Configure Unsafe Trigger (CRITICAL):

Event type: Safe

Trigger: IsSafe=False

Unsafe Delay: Set a delay (e.g., 1-5 minutes) to prevent the dome from closing due to very fast, temporary cloud patches.

Click Add.

When the application detects clouds, it writes IsSafe=False. The ASCOM driver detects this string, reports an UNSAFE status, and your automation software executes the shutdown routine.

🧠 Phase 4: Training Your AI Model (Replication Guide)

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
