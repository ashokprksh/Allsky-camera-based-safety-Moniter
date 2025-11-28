import cv2
import numpy as np
import tensorflow as tf
import os
import time
import re
import paramiko 
import threading
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import json
import socket 

# --- CONFIGURATION DEFAULTS ---
# Using a stable, non-system path for the ASCOM file ensures write permissions.
ASCOM_DEFAULT_PATH = os.path.join(os.path.expanduser("~"), "Documents", "AllSkyMonitor", "ASCOM_STATUS.txt")

DEFAULT_CONFIG = {
    "ASCOM_MONITOR_DELAY": 30, 
    "MODEL_PATH": r'E:\observatory design\Allsky_AI_Training\allsky_cloud_detector_final.tflite',
    "LABELS_PATH": r'E:\observatory design\Allsky_AI_Training\labels.txt',
    "LATEST_IMAGE_PATH": r'E:\observatory design\Allsky_AI_Training\latest.jpg',
    "ALLSKY_HOST": '192.168.1.100',
    "ALLSKY_USER": 'pi',
    "ALLSKY_PASS": 'raspberry',
    "REMOTE_IMAGE_PATH": '/home/pi/allsky/images/latest.jpg',
    "INITIAL_CROP_SIZE": (1300, 1300),
    "SAFE_CONDITIONS": "Clear,Partially Clear,Clear with Moon",
    "SFTP_MAX_RETRIES": 3,
    "SFTP_RETRY_DELAY": 5,
    "ASCOM_FILE_PATH": ASCOM_DEFAULT_PATH, 
}
CONFIG_FILE = "allsky_monitor_config.json"

# AI/Processing Constants
INPUT_SIZE = (224, 224) 

# Global runtime variables
CONFIG = {}
INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
CLASS_NAMES = []

# Global GUI status tracking
CURRENT_STATUS = "STARTING"
CURRENT_CONDITION = "Initializing..."
CURRENT_CONFIDENCE = 0.0
# CRITICAL FIX: Image object is prepared here by the background thread
LATEST_IMAGE_TK = None 
# ---------------------

# --- CONFIGURATION MANAGEMENT ---

def load_config():
    """Loads configuration from JSON file or uses defaults."""
    global CONFIG
    try:
        with open(CONFIG_FILE, 'r') as f:
            CONFIG = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        CONFIG = DEFAULT_CONFIG.copy()
        
    # Ensure a basic directory exists for the ASCOM file if using the default path
    if CONFIG["ASCOM_FILE_PATH"] == ASCOM_DEFAULT_PATH:
        try:
            os.makedirs(os.path.dirname(ASCOM_DEFAULT_PATH), exist_ok=True)
        except:
             pass 

def save_config(new_config):
    """Saves the current configuration to a JSON file."""
    global CONFIG
    CONFIG.update(new_config)
    try:
        # Create the directory for the ASCOM file if it doesn't exist
        os.makedirs(os.path.dirname(CONFIG["ASCOM_FILE_PATH"]), exist_ok=True)
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        return True
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save configuration or create directory: {e}")
        return False

# --- TFLITE AND PREPROCESSING FUNCTIONS ---

def fetch_latest_image_sftp():
    """
    Connects to the Allsky Camera via SFTP and downloads the latest image.
    Implements a retry mechanism for connection stability.
    Returns True on success, False on failure.
    """
    max_retries = CONFIG.get("SFTP_MAX_RETRIES", 3)
    retry_delay = CONFIG.get("SFTP_RETRY_DELAY", 5)

    for attempt in range(max_retries):
        try:
            transport = paramiko.Transport((CONFIG["ALLSKY_HOST"], 22))
            transport.connect(username=CONFIG["ALLSKY_USER"], password=CONFIG["ALLSKY_PASS"])
            sftp = paramiko.SFTPClient.from_transport(transport)

            sftp.get(CONFIG["REMOTE_IMAGE_PATH"], CONFIG["LATEST_IMAGE_PATH"])

            sftp.close()
            transport.close()
            return True 

        except (socket.error, paramiko.SSHException) as e:
            error_msg = str(e)
            print(f"SFTP WARNING: Connection failed on attempt {attempt + 1}. Error: {error_msg}")
            
            if attempt < max_retries - 1:
                print(f"SFTP WARNING: Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("SFTP ERROR: Max retries reached. Failing transfer.")
                return False 

        except paramiko.AuthenticationException:
            print(f"SFTP ERROR: Authentication failed permanently. Check username/password.")
            return False
        
        except Exception as e:
            print(f"SFTP ERROR: Unhandled error: {e}")
            return False

    return False 


def load_model_and_labels():
    """Loads the TFLite Interpreter and class names."""
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES
    
    # Check if necessary paths exist before attempting load
    if not os.path.exists(CONFIG["LABELS_PATH"]) or not os.path.exists(CONFIG["MODEL_PATH"]):
        return False

    try:
        # 1. Load labels
        with open(CONFIG["LABELS_PATH"], 'r') as f:
            lines = f.readlines()
            CLASS_NAMES = [re.sub(r'^\d+\s', '', line.strip()) for line in lines]
            
        # 2. Load TFLite Model
        INTERPRETER = tf.lite.Interpreter(model_path=CONFIG["MODEL_PATH"])
        INTERPRETER.allocate_tensors()
        INPUT_DETAILS = INTERPRETER.get_input_details()
        OUTPUT_DETAILS = INTERPRETER.get_output_details()
        return True

    except Exception as e:
        print(f"FATAL ERROR: Could not load AI model components. Error: {e}")
        global CURRENT_STATUS, CURRENT_CONDITION
        CURRENT_STATUS = "ERROR"
        CURRENT_CONDITION = f"Load Error: {e.__class__.__name__}"
        return False


def preprocess_image_for_prediction(image_path):
    """Performs the exact same preprocessing steps as the training script."""
    try:
        image = cv2.imread(image_path)
        if image is None: raise FileNotFoundError(f"Could not read image from {image_path}")

        h, w = image.shape[:2]
        crop_w, crop_h = CONFIG["INITIAL_CROP_SIZE"]

        # 1. Center Crop 
        if h >= crop_h and w >= crop_w:
            start_x = (w - crop_w) // 2
            start_y = (h - crop_h) // 2
            image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
        
        # 2. Resize to 224x224
        resized_image = cv2.resize(image, INPUT_SIZE, interpolation=cv2.INTER_AREA)

        # 3. Normalize
        normalized_image = (resized_image.astype(np.float32) / 255.0)

        # 4. Reshape
        return np.expand_dims(normalized_image, axis=0)

    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def get_safety_status_ai(image_path):
    """Predicts the sky condition using the TFLite Interpreter."""
    if INTERPRETER is None:
        return False, "ERROR_NO_MODEL", 0.0

    data = preprocess_image_for_prediction(image_path)
    if data is None:
        return False, "ERROR_PREPROCESS", 0.0

    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], data)
    INTERPRETER.invoke()
    
    output_data = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])
    prediction = output_data[0]
    
    index = np.argmax(prediction)
    confidence = float(prediction[index])
    predicted_condition = CLASS_NAMES[index]

    # Check safe conditions (split the comma-separated string from config)
    safe_list = [s.strip() for s in CONFIG["SAFE_CONDITIONS"].split(',')]
    is_safe = predicted_condition in safe_list

    return is_safe, predicted_condition, confidence

# --- MONITORING THREAD LOGIC ---

def prepare_display_image(image_path, app_instance):
    """CRITICAL: Loads, resizes, and prepares the image for Tkinter in the background thread."""
    global LATEST_IMAGE_TK
    try:
        if not os.path.exists(image_path):
            return 
            
        img = Image.open(image_path)
        
        # Target size comes from the GUI instance to match the label size
        display_width = app_instance.image_display_width
        display_height = app_instance.image_display_height
        
        # Resize image while preserving aspect ratio for display
        ratio = min(display_width / img.width, display_height / img.height)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert PIL Image to PhotoImage for Tkinter
        # Note: PhotoImage creation MUST happen after the Tkinter root is created, 
        # but the heavy resizing/loading should be done here.
        # We pass the PIL image back to the main thread for the final PhotoImage conversion.
        return img 

    except Exception as e:
        print(f"ERROR: Could not prepare image for display: {e}")
        return None


def monitor_loop(app_instance):
    """The background thread that runs the SFTP, AI, and file write tasks."""
    global CURRENT_STATUS, CURRENT_CONDITION, CURRENT_CONFIDENCE, LATEST_IMAGE_TK
    
    if not load_model_and_labels():
        CURRENT_STATUS = "ERROR"
        CURRENT_CONDITION = "Model Load Failed"
    
    print("--- Starting All-Sky Safety Monitor ---")

    while True:
        try:
            if INTERPRETER is None:
                if load_model_and_labels():
                    print("Model reloaded successfully after configuration update.")
                else:
                    time.sleep(CONFIG["ASCOM_MONITOR_DELAY"])
                    continue

            image_fetched = fetch_latest_image_sftp()
            
            if not image_fetched:
                is_safe = False
                condition = "Transfer Error"
                confidence = 1.0
            
            elif not os.path.exists(CONFIG["LATEST_IMAGE_PATH"]):
                 is_safe = False
                 condition = "Image Missing"
                 confidence = 1.0
            
            else:
                # 1. Run AI prediction
                is_safe, condition, confidence = get_safety_status_ai(CONFIG["LATEST_IMAGE_PATH"])

                # 2. Prepare image for GUI (Heavy lifting done in background)
                display_img_pil = prepare_display_image(CONFIG["LATEST_IMAGE_PATH"], app_instance)
                if display_img_pil:
                    # Conversion to PhotoImage must be done *in* the main thread
                    # We use a trick by running the PhotoImage conversion via 'after'
                    app_instance.after(0, app_instance.update_image_display_thread_safe, display_img_pil)
            
            # Update global variables for GUI thread to read
            CURRENT_STATUS = "SAFE" if is_safe else "UNSAFE"
            CURRENT_CONDITION = condition
            CURRENT_CONFIDENCE = confidence
            
            # --- ASCOM Integration Point (Write Status File) ---
            ascom_file_path = CONFIG["ASCOM_FILE_PATH"]

            os.makedirs(os.path.dirname(ascom_file_path), exist_ok=True)

            with open(ascom_file_path, 'w') as f:
                f.write(f"IsSafe={is_safe}\n")
                f.write(f"Condition={condition}\n")
                f.write(f"Confidence={confidence:.2f}\n")
            
            if not is_safe:
                print(f"[{time.strftime('%H:%M:%S')}] WARNING: {condition} detected. Status: UNSAFE.")
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Status: SAFE | Condition: {condition} (Conf: {confidence:.2f})")
            
            # --- CRITICAL: Tell the GUI to update its status labels/indicators ---
            if app_instance.winfo_exists():
                app_instance.after(0, app_instance.trigger_gui_refresh)
            # -----------------------------------------------------------------------------------

        except Exception as e:
            print(f"Unexpected error in monitor loop: {e}")
            CURRENT_STATUS = "ERROR"
            CURRENT_CONDITION = "Runtime Error"
            CURRENT_CONFIDENCE = 0.0

        time.sleep(CONFIG["ASCOM_MONITOR_DELAY"])

# --- SETTINGS WINDOW CLASS ---

class SettingsWindow(tk.Toplevel):
    # ... (SettingsWindow code remains unchanged)
    def __init__(self, master, current_config):
        super().__init__(master)
        self.title("Configuration Settings")
        self.config = current_config.copy() 
        self.transient(master) # Make sure window stays on top
        self.grab_set()
        self.resizable(False, False)
        
        self.frame = ttk.Frame(self, padding="15")
        self.frame.pack(fill="both", expand=True)
        
        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        # Helper to create label and entry field
        def create_field(row, label_text, key, file_dialog=False):
            ttk.Label(self.frame, text=label_text + ":").grid(row=row, column=0, sticky='w', pady=5, padx=5)
            entry = ttk.Entry(self.frame, width=50)
            entry.insert(0, str(self.config.get(key, '')))
            entry.grid(row=row, column=1, sticky='ew', pady=5, padx=5)
            self.entries[key] = entry
            
            if file_dialog:
                ttk.Button(self.frame, text="Browse", command=lambda: self.browse_file(key)).grid(row=row, column=2, padx=5)

        row = 0
        # --- ASCOM Output Path ---
        ttk.Label(self.frame, text="--- ASCOM OUTPUT FILE ---", font=('Inter', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=(10, 5))
        row += 1
        create_field(row, "ASCOM Status File Path", "ASCOM_FILE_PATH", file_dialog=True)
        row += 1
        
        # --- AI Model Paths ---
        ttk.Label(self.frame, text="--- AI Model Paths ---", font=('Inter', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=(10, 5))
        row += 1
        create_field(row, "Model File (.tflite)", "MODEL_PATH", file_dialog=True)
        row += 1
        create_field(row, "Labels File (.txt)", "LABELS_PATH", file_dialog=True)
        row += 1
        
        # --- Image Handling ---
        ttk.Label(self.frame, text="--- Image Handling ---", font=('Inter', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=(10, 5))
        row += 1
        create_field(row, "Local Image Path (Save)", "LATEST_IMAGE_PATH", file_dialog=True)
        row += 1
        
        ttk.Label(self.frame, text="Initial Crop (e.g., 1300,1300):").grid(row=row, column=0, sticky='w', pady=5, padx=5)
        crop_entry = ttk.Entry(self.frame, width=20)
        crop_entry.insert(0, ",".join(map(str, self.config.get("INITIAL_CROP_SIZE", (1300, 1300)))))
        crop_entry.grid(row=row, column=1, sticky='w', pady=5, padx=5)
        self.entries["INITIAL_CROP_SIZE"] = crop_entry
        row += 1

        # --- SFTP Connection ---
        ttk.Label(self.frame, text="--- SFTP Connection ---", font=('Inter', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=(10, 5))
        row += 1
        create_field(row, "Allsky IP/Host", "ALLSKY_HOST")
        row += 1
        create_field(row, "SFTP Username", "ALLSKY_USER")
        row += 1
        create_field(row, "SFTP Password", "ALLSKY_PASS")
        row += 1
        create_field(row, "Remote Image Path (Pi)", "REMOTE_IMAGE_PATH")
        row += 1

        # --- Safety Logic & Timing ---
        ttk.Label(self.frame, text="--- Safety Logic & Timing ---", font=('Inter', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky='w', pady=(10, 5))
        row += 1
        create_field(row, "Monitor Delay (s)", "ASCOM_MONITOR_DELAY")
        row += 1
        create_field(row, "Safe Conditions (Comma Sep)", "SAFE_CONDITIONS")
        row += 1

        # --- Buttons ---
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=row, column=0, columnspan=3, pady=15)
        
        ttk.Button(button_frame, text="Save & Restart", command=self.save_and_exit).pack(side='left', padx=10)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side='left', padx=10)

    def browse_file(self, key):
        if key.endswith("PATH"):
            if key in ["MODEL_PATH", "LABELS_PATH"]:
                file_path = filedialog.askopenfilename()
            else:
                file_path = filedialog.asksaveasfilename(defaultextension=".txt" if key == "ASCOM_FILE_PATH" else ".jpg", 
                                                         filetypes=[("Text files", "*.txt")] if key == "ASCOM_FILE_PATH" else [("JPEG files", "*.jpg")])
        
            if file_path:
                self.entries[key].delete(0, tk.END)
                self.entries[key].insert(0, file_path)
    
    def save_and_exit(self):
        new_config = {}
        try:
            for key, entry in self.entries.items():
                if key not in ["INITIAL_CROP_SIZE", "ASCOM_MONITOR_DELAY"]:
                    new_config[key] = entry.get()

            new_config["ASCOM_MONITOR_DELAY"] = int(self.entries["ASCOM_MONITOR_DELAY"].get())
            
            crop_str = self.entries["INITIAL_CROP_SIZE"].get().split(',')
            new_config["INITIAL_CROP_SIZE"] = (int(crop_str[0].strip()), int(crop_str[1].strip()))

        except ValueError:
            messagebox.showerror("Input Error", "Please ensure 'Monitor Delay' and 'Initial Crop' are valid numbers.")
            return

        if save_config(new_config):
            messagebox.showinfo("Success", "Configuration saved. The monitor thread will restart with new settings.")
            global INTERPRETER
            INTERPRETER = None 
            self.master.winfo_exists() and self.master.update()
            self.destroy()

# --- GUI APPLICATION CLASS ---

class AllSkyMonitorApp(tk.Tk):
    def __init__(self):
        load_config() 
        
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.title("All-Sky AI Safety Monitor")
        self.geometry("700x550")
        self.configure(bg="#222222")

        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure('TLabel', background='#222222', foreground='white', font=('Inter', 12))
        self.style.configure('Title.TLabel', font=('Inter', 18, 'bold'))
        self.style.configure('Settings.TButton', font=('Inter', 10, 'bold'), foreground='#222222')

        self.image_display_width = 450
        self.image_display_height = 450
        self.is_running = True

        self.create_widgets()
        
        self.monitor_thread = threading.Thread(target=monitor_loop, args=(self,), daemon=True)
        self.monitor_thread.start()

        self.update_gui()

    def create_widgets(self):
        self.main_frame = ttk.Frame(self, padding="10", style='TFrame')
        self.main_frame.pack(fill='both', expand=True)
        self.main_frame.grid_columnconfigure(0, weight=0) 
        self.main_frame.grid_columnconfigure(1, weight=1) 
        
        # --- Status Panel (Column 0) ---
        self.status_frame = ttk.Frame(self.main_frame, padding="10", style='TFrame')
        self.status_frame.grid(row=0, column=0, sticky='nswe', padx=10, pady=10)
        
        ttk.Label(self.status_frame, text="AI Safety Status", style='Title.TLabel').pack(pady=10)
        
        ttk.Button(self.status_frame, text="⚙️ Settings", command=self.open_settings, style='Settings.TButton').pack(pady=(0, 20))

        self.safety_canvas = tk.Canvas(self.status_frame, width=150, height=150, bg="#222222", highlightthickness=0)
        self.safety_canvas.pack(pady=10)
        self.safety_indicator = self.safety_canvas.create_oval(10, 10, 140, 140, fill="gray")
        
        ttk.Label(self.status_frame, text="Condition:", style='TLabel').pack(pady=(15, 0))
        self.condition_label = ttk.Label(self.status_frame, text="Waiting...", font=('Inter', 14, 'bold'))
        self.condition_label.pack(pady=5)
        
        ttk.Label(self.status_frame, text="Confidence:", style='TLabel').pack(pady=(15, 0))
        self.confidence_label = ttk.Label(self.status_frame, text="0.00", font=('Inter', 12))
        self.confidence_label.pack(pady=5)

        ttk.Label(self.status_frame, text="ASCOM File:", style='TLabel').pack(pady=(20, 0))
        self.file_status_label = ttk.Label(self.status_frame, text=os.path.basename(CONFIG["ASCOM_FILE_PATH"]), font=('Inter', 8))
        self.file_status_label.pack(pady=5)

        # --- Image Panel (Column 1) ---
        ttk.Label(self.main_frame, text="Latest All-Sky Image (Live)", style='Title.TLabel').grid(row=0, column=1, sticky='n', pady=10, padx=10)

        self.image_label = ttk.Label(self.main_frame, text="Loading image...")
        self.image_label.grid(row=0, column=1, sticky='n', padx=10, pady=(50, 10))

    def open_settings(self):
        if not hasattr(self, '_settings_window') or not self._settings_window.winfo_exists():
            self._settings_window = SettingsWindow(self, CONFIG)
            self.wait_window(self._settings_window)

    def trigger_gui_refresh(self):
        """Called by the background thread to force the main thread to update the status labels."""
        self.update_status_labels()

    def update_image_display_thread_safe(self, pil_image):
        """Final conversion to PhotoImage and display, runs in the main thread."""
        try:
            # Conversion to PhotoImage MUST be in the main thread
            self.latest_image_tk = ImageTk.PhotoImage(pil_image)
            
            # Update the image label
            self.image_label.config(image=self.latest_image_tk, text="")
            self.image_label.image = self.latest_image_tk 
        except Exception as e:
            print(f"Error updating image display in main thread: {e}")


    def update_status_labels(self):
        """Updates only the status text labels and indicator."""
        # Update Status Indicator
        if CURRENT_STATUS == "SAFE":
            color = "#4CAF50" 
        elif CURRENT_STATUS == "UNSAFE":
            color = "#F44336" 
        elif CURRENT_STATUS == "ERROR":
            color = "#FF9800" 
        else:
            color = "gray" 

        self.safety_canvas.itemconfig(self.safety_indicator, fill=color)
        
        # Update text labels
        self.condition_label.config(text=CURRENT_CONDITION.upper())
        self.confidence_label.config(text=f"{CURRENT_CONFIDENCE:.2f}")
        
        # Update file path label in case the settings changed
        self.file_status_label.config(text=os.path.basename(CONFIG["ASCOM_FILE_PATH"]))


    def update_gui(self):
        """Main GUI loop clock. Runs every 1000ms."""
        self.update_status_labels() # Keep status labels updated frequently

        if self.is_running:
            self.after(1000, self.update_gui)

    def on_closing(self):
        self.is_running = False
        messagebox.showinfo("Monitoring Stopped", "The background monitor thread has been stopped.")
        self.destroy()

if __name__ == '__main__':
    try:
        import PIL.Image
        import PIL.ImageTk
    except ImportError:
        print("\nFATAL ERROR: PIL/Pillow not installed. Please run: pip install Pillow")
        exit()
    try:
        import paramiko
    except ImportError:
        print("\nFATAL ERROR: paramiko not installed. Please run: pip install paramiko")
        exit()
    
    app = AllSkyMonitorApp()
    app.mainloop()