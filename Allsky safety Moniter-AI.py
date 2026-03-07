import cv2
import numpy as np
import tensorflow as tf
import os
import time
import re
import paramiko 
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import socket 
import math
import requests 
import csv
import base64
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from urllib.parse import parse_qs, urlparse

# Import MQTT with version-aware error handling for Paho v2.0+
try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
    MQTT_V2 = True
except ImportError:
    try:
        import paho.mqtt.client as mqtt
        MQTT_V2 = False
    except ImportError:
        mqtt = None
        MQTT_V2 = False

# --- CONFIGURATION DEFAULTS ---
ASCOM_DEFAULT_PATH = os.path.join(os.path.expanduser("~"), "Documents", "AllSkyMonitor", "ASCOM_STATUS.txt")
LOG_DEFAULT_PATH = os.path.join(os.path.expanduser("~"), "Documents", "AllSkyMonitor", "weather_log.csv")

DEFAULT_CONFIG = {
    "ASCOM_MONITOR_DELAY": 30, 
    "FETCH_METHOD": "MQTT", 
    "MODEL_PATH": "",
    "LABELS_PATH": "",
    "LATEST_IMAGE_PATH": "latest.jpg",
    "ALLSKY_HOST": '192.168.1.100',
    "ALLSKY_USER": 'pi',
    "ALLSKY_PASS": 'raspberry',
    "REMOTE_IMAGE_PATH": '/home/pi/allsky/images/latest.jpg',
    "SFTP_MAX_RETRIES": 3,
    "SFTP_RETRY_DELAY": 5,
    "MQTT_BROKER": "192.168.1.50", 
    "MQTT_PORT": 1883,
    "MQTT_USER": "", 
    "MQTT_PASS": "",
    "MQTT_TOPIC": "allsky/image",
    "IP_CAM_URL": "", 
    "INITIAL_CROP_SIZE_W": 1300,
    "INITIAL_CROP_SIZE_H": 1300,
    "SAFE_CONDITIONS": "Clear,Partially Clear,Clear with Moon",
    "ASCOM_FILE_PATH": ASCOM_DEFAULT_PATH,
    "LOG_FILE_PATH": LOG_DEFAULT_PATH,
    "USE_WEATHER_STATION": True,
    "ECOWITT_PORT": 8080,
    "WIND_LIMIT": 30.0,
    "HUMIDITY_LIMIT": 90.0,
    "CLOUD_GRACE_MINS": 5,
    "FORWARD_DATA": False,
    "FORWARD_URL": "",
    "USE_WEB_SERVER": True,
    "WEB_SERVER_PORT": 8000,
    "ALPACA_DEVICE_NUMBER": 0
}
CONFIG_FILE = "allsky_monitor_config_ai.json"

# Global State Variables
CONFIG = {}
INTERPRETER = None
INPUT_DETAILS = None
OUTPUT_DETAILS = None
CLASS_NAMES = []
CURRENT_STATUS = "STARTING"
CURRENT_CONDITION = "Initializing AI..."
CURRENT_CONFIDENCE = 0.0
LAST_IMAGE_TIME = "Never"
MQTT_CONN_STATUS = "Disconnected"
IS_REALLY_SAFE_GLOBAL = False 
SERVER_TRANSACTION_ID = 0
LAST_SAFE_TIME = None 
IS_IN_GRACE_PERIOD = False

# Weather Data
WEATHER_DATA = {
    "temp": 0.0, "humidity": 0, "pressure": 0.0, 
    "wind_speed": 0.0, "wind_gust": 0.0, "wind_dir": 0, 
    "rain_rate": 0.0, "is_raining": False, 
    "solar": 0, "uv": 0, "last_update": 0
}
WEATHER_HISTORY = [] 
HISTORY_LIMIT = 1500 

# --- THREADING SERVER ---
class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True

# --- CORE UTILITIES ---

def load_config():
    global CONFIG
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                CONFIG = json.load(f)
        else:
            CONFIG = DEFAULT_CONFIG.copy()
        for k, v in DEFAULT_CONFIG.items():
            if k not in CONFIG:
                CONFIG[k] = v
    except Exception:
        CONFIG = DEFAULT_CONFIG.copy()

def save_config(new_config):
    global CONFIG
    CONFIG.update(new_config)
    try:
        os.makedirs(os.path.dirname(CONFIG["ASCOM_FILE_PATH"]), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(CONFIG, f, indent=4)
        return True
    except Exception as e:
        messagebox.showerror("Save Error", f"Could not save config: {e}")
        return False

# --- INTERNAL GRAPHING ENGINE ---

class MiniGraph(tk.Canvas):
    def __init__(self, master, title, color, **kwargs):
        super().__init__(master, bg='#1a1a1a', highlightthickness=1, highlightbackground='#333', **kwargs)
        self.title, self.color, self.data = title, color, []
    
    def update_data(self, history_index):
        if not WEATHER_HISTORY: return
        self.data = [h[history_index] for h in WEATHER_HISTORY]
        self.draw()

    def draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or not self.data: return
        m_l, m_r, m_t, m_b = 40, 10, 20, 10
        gw, gh = w - m_l - m_r, h - m_t - m_b
        mi, ma = min(self.data), max(self.data)
        if ma == mi: ma += 0.1
        v_min, v_max = mi - (ma-mi)*0.1, ma + (ma-mi)*0.1
        
        self.create_line(m_l, m_t, m_l, m_t+gh, fill='#333')
        self.create_line(m_l, m_t+gh, w-m_r, m_t+gh, fill='#333')
        
        self.create_text(m_l-5, m_t, text=f"{ma:.1f}", fill='#888', anchor='ne', font=('Arial', 7))
        self.create_text(m_l-5, m_t+gh, text=f"{mi:.1f}", fill='#888', anchor='ne', font=('Arial', 7))
        self.create_text(5, 5, text=self.title, fill='white', anchor='nw', font=('Segoe UI', 8, 'bold'))
        self.create_text(w-5, 5, text=f"{self.data[-1]:.1f}", fill=self.color, anchor='ne', font=('Segoe UI', 8, 'bold'))

        points = []
        x_step = gw / max(len(self.data)-1, 1)
        for i, val in enumerate(self.data):
            x = m_l + (i * x_step)
            y = m_t + gh - ((val - v_min) / (v_max - v_min) * gh)
            points.extend([x, y])
        if len(points) >= 4:
            self.create_line(points, fill=self.color, width=2, smooth=True)

class CompassWidget(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, width=140, height=140, bg='#121212', highlightthickness=0, **kwargs)
    
    def update_dir(self, angle):
        self.delete("all")
        cx, cy = 70, 70
        r = 60
        self.create_oval(cx-r, cy-r, cx+r, cy+r, outline='#555', width=3)
        for i, t in enumerate(['N','E','S','W']):
            ang = math.radians(i*90 - 90)
            tx, ty = cx + (r-15)*math.cos(ang), cy + (r-15)*math.sin(ang)
            self.create_text(tx, ty, text=t, fill='#AAA', font=('Arial', 10, 'bold'))
        
        rad = math.radians(angle - 90)
        nx, ny = cx + (r-5)*math.cos(rad), cy + (r-5)*math.sin(rad)
        tx, ty = cx - (r-40)*math.cos(rad), cy - (r-40)*math.sin(rad)
        self.create_line(tx, ty, nx, ny, fill='#FF5722', width=4, arrow=tk.LAST, arrowshape=(16,20,6))
        self.create_oval(cx-4, cy-4, cx+4, cy+4, fill='#FFF')
        self.create_text(cx, cy+r+10, text=f"{angle}°", fill='white', font=('Arial', 9))

# --- SERVER HANDLERS ---

class AlpacaDashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return

    def _get_alpaca_params(self, body=None):
        params = {}
        parsed_url = urlparse(self.path)
        qs = parse_qs(parsed_url.query)
        for k, v in qs.items(): params[k.lower()] = v[0]
        if body:
            post_data = parse_qs(body.decode('utf-8'))
            for k, v in post_data.items(): params[k.lower()] = v[0]
        return params

    def _send_alpaca_resp(self, client_id, value=None, error_num=0, error_msg="", status=200):
        global SERVER_TRANSACTION_ID
        SERVER_TRANSACTION_ID += 1
        try: clean_id = int(client_id)
        except: clean_id = 0
        response = { "ClientTransactionID": clean_id, "ServerTransactionID": SERVER_TRANSACTION_ID, "ErrorNumber": error_num, "ErrorMessage": error_msg }
        if value is not None: response["Value"] = value
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def do_PUT(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else None
            params = self._get_alpaca_params(body)
            client_id = params.get('clienttransactionid', 0)
            path_parts = [p.lower() for p in urlparse(self.path).path.split('/') if p]
            
            if len(path_parts) >= 5 and path_parts[0] == 'api' and path_parts[1] == 'v1' and path_parts[2] == 'safetymonitor':
                try:
                    if int(path_parts[3]) != int(CONFIG["ALPACA_DEVICE_NUMBER"]):
                        return self._send_alpaca_resp(client_id, error_num=1025, error_msg="Device mismatch", status=400)
                except: return self._send_alpaca_resp(client_id, error_num=1025, error_msg="Bad device ID", status=400)
                
                if path_parts[4] == 'connected':
                    if params.get('connected', '').lower() not in ['true', 'false']:
                        return self._send_alpaca_resp(client_id, error_num=1025, error_msg="Invalid Boolean", status=400)
                    return self._send_alpaca_resp(client_id)
            self.send_error(404)
        except: self.send_error(500)

    def do_GET(self):
        try:
            parsed_url = urlparse(self.path)
            path_parts = [p.lower() for p in parsed_url.path.split('/') if p]
            params = self._get_alpaca_params()
            client_id = params.get('clienttransactionid', 0)

            if len(path_parts) >= 2 and path_parts[0] == 'management':
                if path_parts[1] == 'v1' and path_parts[2] == 'configureddevices':
                    return self._send_alpaca_resp(client_id, value=[{"DeviceName": "AI Safety Monitor", "DeviceType": "SafetyMonitor", "DeviceNumber": int(CONFIG["ALPACA_DEVICE_NUMBER"]), "UniqueID": "AI-SAFE-MON"}])
                if path_parts[2] == 'apiversions': return self._send_alpaca_resp(client_id, value=[1])
            
            if len(path_parts) >= 5 and path_parts[0] == 'api' and path_parts[1] == 'v1' and path_parts[2] == 'safetymonitor':
                try:
                    if int(path_parts[3]) != int(CONFIG["ALPACA_DEVICE_NUMBER"]):
                        return self._send_alpaca_resp(client_id, error_num=1025, error_msg="Device mismatch", status=400)
                except: return self._send_alpaca_resp(client_id, error_num=1025, error_msg="Bad device ID", status=400)
                
                cmd = path_parts[4]
                if cmd == 'issafe': return self._send_alpaca_resp(client_id, value=IS_REALLY_SAFE_GLOBAL)
                elif cmd == 'connected': return self._send_alpaca_resp(client_id, value=True)
                elif cmd == 'name': return self._send_alpaca_resp(client_id, value="Observatory Guardian AI")
                elif cmd == 'driverinfo': return self._send_alpaca_resp(client_id, value="AI Safety Monitor v5.0")
                elif cmd == 'driverversion': return self._send_alpaca_resp(client_id, value="5.0")
                elif cmd == 'interfaceversion': return self._send_alpaca_resp(client_id, value=1)
                return self.send_error(404)

            if parsed_url.path.lower() in ['/', '/index.html']:
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                
                step = 10 if len(WEATHER_HISTORY) > 100 else 1
                times = [h[0].strftime("%H:%M") for h in WEATHER_HISTORY[::step]]
                d_temp = [h[1] for h in WEATHER_HISTORY[::step]]
                d_hum = [h[2] for h in WEATHER_HISTORY[::step]]
                d_press = [h[3] for h in WEATHER_HISTORY[::step]]
                d_wind = [h[4] for h in WEATHER_HISTORY[::step]]
                d_gust = [h[5] for h in WEATHER_HISTORY[::step]]
                d_rain = [h[6] for h in WEATHER_HISTORY[::step]]
                d_solar = [h[7] for h in WEATHER_HISTORY[::step]]
                d_uv = [h[8] for h in WEATHER_HISTORY[::step]]

                html = f"""
                <!DOCTYPE html><html><head><title>Observatory Guardian AI</title>
                <meta http-equiv='refresh' content='60'>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {{ font-family: 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; text-align: center; padding: 15px; margin: 0; }}
                    .status-bar {{ font-size: 20px; font-weight: 800; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 2px solid; }}
                    .SAFE {{ background: #1b5e20; color: #81c784; border-color: #4caf50; }} 
                    .UNSAFE {{ background: #b71c1c; color: #ef9a9a; border-color: #f44336; }}
                    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; max-width: 1600px; margin: 0 auto; }}
                    .card {{ background: #161616; padding: 15px; border-radius: 12px; border: 1px solid #333; }}
                    .compass {{ width: 120px; height: 120px; border-radius: 50%; border: 4px solid #444; position: relative; margin: 0 auto; background: #222; }}
                    .arrow {{ width: 4px; height: 50px; background: #FF5722; position: absolute; top: 50%; left: 50%; transform-origin: top center; transition: transform 0.5s ease; }}
                    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }}
                    td {{ padding: 6px; border-bottom: 1px solid #333; text-align: left; }}
                    td:last-child {{ text-align: right; font-weight: bold; color: #aaa; }}
                    h3 {{ margin-top: 0; color: #ccc; border-bottom: 1px solid #333; padding-bottom: 10px; }}
                </style>
                </head><body>
                <div class="status-bar {CURRENT_STATUS}">{CURRENT_STATUS} - {CURRENT_CONDITION} ({CURRENT_CONFIDENCE*100:.1f}%)</div>
                
                <div class="grid">
                    <div class="card">
                        <h3>Sky Cam AI</h3>
                        <img src="/latest.jpg?t={int(time.time())}" style="max-width: 100%; border-radius: 6px;">
                        <p>AI Prediction: {CURRENT_CONDITION} | Confidence: {CURRENT_CONFIDENCE*100:.1f}%</p>
                    </div>
                    <div class="card">
                        <h3>Live Conditions</h3>
                        <div style="display: flex; align-items: center; justify-content: center; gap: 20px;">
                            <div class="compass">
                                <div class="arrow" style="transform: rotate({WEATHER_DATA['wind_dir'] + 180}deg) translateY(0);"></div>
                                <div style="position:absolute; top:5px; left:52px; font-size:12px; font-weight:bold;">N</div>
                                <div style="position:absolute; bottom:-25px; width:100%; text-align:center;">{WEATHER_DATA['wind_dir']}&deg;</div>
                            </div>
                            <div style="flex-grow: 1;">
                                <table>
                                    <tr><td>Temperature</td><td>{WEATHER_DATA['temp']} &deg;C</td></tr>
                                    <tr><td>Humidity</td><td>{WEATHER_DATA['humidity']} %</td></tr>
                                    <tr><td>Pressure</td><td>{WEATHER_DATA['pressure']} hPa</td></tr>
                                    <tr><td>Wind Speed</td><td>{WEATHER_DATA['wind_speed']:.1f} km/h</td></tr>
                                    <tr><td>Wind Gust</td><td>{WEATHER_DATA['wind_gust']:.1f} km/h</td></tr>
                                    <tr><td>Rain Rate</td><td>{WEATHER_DATA['rain_rate']} in/hr</td></tr>
                                    <tr><td>Solar Rad</td><td>{WEATHER_DATA['solar']} W/m&sup2;</td></tr>
                                    <tr><td>UV Index</td><td>{WEATHER_DATA['uv']}</td></tr>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="grid">
                    <div class="card"><canvas id="cTemp"></canvas></div>
                    <div class="card"><canvas id="cHum"></canvas></div>
                    <div class="card"><canvas id="cPress"></canvas></div>
                    <div class="card"><canvas id="cWind"></canvas></div>
                    <div class="card"><canvas id="cRain"></canvas></div>
                    <div class="card"><canvas id="cSolar"></canvas></div>
                </div>

                <script>
                    const times = {json.dumps(times)};
                    const common = {{ 
                        type: 'line', 
                        options: {{ 
                            responsive: true,
                            plugins: {{ legend: {{ display: false }} }}, 
                            scales: {{ 
                                x: {{ display: false }}, 
                                y: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#888' }}, title: {{ display: true, color: '#666' }} }} 
                            }} 
                        }} 
                    }};
                    
                    const createChart = (id, label, data, color, unit) => {{
                        const cfg = JSON.parse(JSON.stringify(common));
                        cfg.data = {{ labels: times, datasets: [{{ label: label, data: data, borderColor: color, backgroundColor: color+'20', fill: true, tension: 0.3, pointRadius: 0 }}] }};
                        cfg.options.plugins.title = {{ display: true, text: label, color: '#ccc', font: {{ size: 16 }} }};
                        cfg.options.scales.y.title.text = unit;
                        new Chart(document.getElementById(id), cfg);
                    }};

                    createChart('cTemp', 'Temperature', {json.dumps(d_temp)}, '#FF5722', '°C');
                    createChart('cHum', 'Humidity', {json.dumps(d_hum)}, '#2196F3', '%');
                    createChart('cPress', 'Pressure', {json.dumps(d_press)}, '#9C27B0', 'hPa');
                    createChart('cRain', 'Rain Rate', {json.dumps(d_rain)}, '#03A9F4', 'in/hr');
                    createChart('cSolar', 'Solar / UV', {json.dumps(d_solar)}, '#FFC107', 'W/m²');
                    
                    const wCfg = JSON.parse(JSON.stringify(common));
                    wCfg.data = {{ labels: times, datasets: [
                        {{ label: 'Speed', data: {json.dumps(d_wind)}, borderColor: '#4CAF50', tension: 0.3, pointRadius: 0 }},
                        {{ label: 'Gust', data: {json.dumps(d_gust)}, borderColor: '#81C784', borderDash: [5,5], pointRadius: 0 }}
                    ]}};
                    wCfg.options.plugins.title = {{ display: true, text: 'Wind', color: '#ccc', font: {{ size: 16 }} }};
                    wCfg.options.scales.y.title.text = 'km/h';
                    wCfg.options.plugins.legend.display = true;
                    new Chart(document.getElementById('cWind'), wCfg);
                </script>
                </body></html>"""
                self.wfile.write(html.encode())
            elif parsed_url.path.startswith('/latest.jpg'):
                if os.path.exists(CONFIG["LATEST_IMAGE_PATH"]):
                    self.send_response(200); self.send_header('Content-type', 'image/jpeg'); self.end_headers()
                    with open(CONFIG["LATEST_IMAGE_PATH"], 'rb') as f: self.wfile.write(f.read())
        except Exception: pass

class EcowittHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            raw_post_data = self.rfile.read(content_length)
            data = parse_qs(raw_post_data.decode('utf-8'))
            global WEATHER_DATA, WEATHER_HISTORY
            
            temp = round((float(data.get('tempf', [0])[0]) - 32) * 5/9, 1)
            hum = int(data.get('humidity', [0])[0])
            press = round(float(data.get('baromrelin', [29.92])[0]) * 33.8639, 1)
            ws = float(data.get('windspeedmph', [0])[0]) * 1.60934
            wg = float(data.get('windgustmph', [0])[0]) * 1.60934
            wdir = int(data.get('winddir', [0])[0])
            rain = float(data.get('rainratein', [0])[0])
            solar = float(data.get('solarradiation', [0])[0])
            uv = int(data.get('uv', [0])[0])
            
            WEATHER_DATA.update({
                "temp": temp, "humidity": hum, "pressure": press,
                "wind_speed": ws, "wind_gust": wg, "wind_dir": wdir,
                "rain_rate": rain, "is_raining": rain > 0,
                "solar": solar, "uv": uv, "last_update": time.time()
            })
            
            WEATHER_HISTORY.append((datetime.now(), temp, hum, press, ws, wg, rain, solar, uv))
            if len(WEATHER_HISTORY) > HISTORY_LIMIT: WEATHER_HISTORY.pop(0)

            if CONFIG.get("FORWARD_DATA") and CONFIG.get("FORWARD_URL"):
                threading.Thread(target=lambda: requests.post(CONFIG["FORWARD_URL"], data=raw_post_data, timeout=5)).start()
            
            self.send_response(200); self.end_headers()
        except: self.send_response(500); self.end_headers()

# --- DISCOVERY & MQTT ---

def run_alpaca_discovery_responder():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try: sock.bind(('', 32227))
    except: return
    while True:
        try:
            data, addr = sock.recvfrom(1024)
            if "alpacadiscovery1" in data.decode('utf-8'):
                resp = json.dumps({"AlpacaPort": int(CONFIG["WEB_SERVER_PORT"])})
                sock.sendto(resp.encode('utf-8'), addr)
        except: time.sleep(2)

def fetch_latest_image_sftp():
    try:
        transport = paramiko.Transport((CONFIG["ALLSKY_HOST"], 22))
        transport.connect(username=CONFIG["ALLSKY_USER"], password=CONFIG["ALLSKY_PASS"])
        sftp = paramiko.SFTPClient.from_transport(transport)
        sftp.get(CONFIG["REMOTE_IMAGE_PATH"], CONFIG["LATEST_IMAGE_PATH"])
        sftp.close(); transport.close(); return True 
    except: return False

def fetch_latest_image_ipcam():
    url = CONFIG.get("IP_CAM_URL", "")
    if not url: return False
    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened(): return False
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(CONFIG["LATEST_IMAGE_PATH"], frame)
            global LAST_IMAGE_TIME
            LAST_IMAGE_TIME = datetime.now().strftime("%H:%M:%S")
        cap.release(); return ret
    except: return False

def on_mqtt_message(client, userdata, message):
    global LAST_IMAGE_TIME
    try:
        payload = message.payload
        if payload.startswith(b'\xff\xd8') or payload.startswith(b'\x89PNG'):
            with open(CONFIG["LATEST_IMAGE_PATH"], "wb") as f: f.write(payload)
            LAST_IMAGE_TIME = datetime.now().strftime("%H:%M:%S")
    except: pass

def run_mqtt_listener():
    global MQTT_CONN_STATUS
    if mqtt is None: return
    client = mqtt.Client(CallbackAPIVersion.VERSION1) if MQTT_V2 else mqtt.Client()
    if CONFIG.get("MQTT_USER"): client.username_pw_set(CONFIG["MQTT_USER"], CONFIG.get("MQTT_PASS", ""))
    
    def on_connect(c, u, f, rc):
        global MQTT_CONN_STATUS
        if rc == 0:
            MQTT_CONN_STATUS = "Connected"
            client.subscribe(CONFIG["MQTT_TOPIC"])
        else:
            MQTT_CONN_STATUS = f"Auth Fail ({rc})"
    
    def on_disconnect(c, u, rc):
        global MQTT_CONN_STATUS
        MQTT_CONN_STATUS = "Disconnected"

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_mqtt_message
    
    while True:
        try: 
            client.connect(CONFIG["MQTT_BROKER"], int(CONFIG["MQTT_PORT"]), 60)
            client.loop_forever()
        except:
            MQTT_CONN_STATUS = "Net Error"
            time.sleep(10)

# --- AI ENGINE ---

def try_load_ai():
    global INTERPRETER, INPUT_DETAILS, OUTPUT_DETAILS, CLASS_NAMES
    try:
        if not os.path.exists(CONFIG["LABELS_PATH"]) or not os.path.exists(CONFIG["MODEL_PATH"]):
            return False
        with open(CONFIG["LABELS_PATH"], 'r') as f:
            CLASS_NAMES = [re.sub(r'^\d+\s', '', l.strip()) for l in f.readlines()]
        INTERPRETER = tf.lite.Interpreter(model_path=CONFIG["MODEL_PATH"])
        INTERPRETER.allocate_tensors()
        INPUT_DETAILS = INTERPRETER.get_input_details()
        OUTPUT_DETAILS = INTERPRETER.get_output_details()
        return True
    except Exception:
        return False

def monitor_loop(app_instance):
    global CURRENT_STATUS, CURRENT_CONDITION, CURRENT_CONFIDENCE
    global IS_REALLY_SAFE_GLOBAL, LAST_SAFE_TIME, IS_IN_GRACE_PERIOD
    
    while True:
        try:
            if INTERPRETER is None:
                if not try_load_ai():
                    CURRENT_STATUS, CURRENT_CONDITION = "ERROR", "AI Model Missing"
                    app_instance.after(0, app_instance.trigger_gui_refresh)
                    time.sleep(CONFIG["ASCOM_MONITOR_DELAY"]); continue

            image_fresh = False
            method = CONFIG.get("FETCH_METHOD", "MQTT")
            if method == "SFTP": image_fresh = fetch_latest_image_sftp()
            elif method == "IP_CAM": image_fresh = fetch_latest_image_ipcam()
            elif os.path.exists(CONFIG["LATEST_IMAGE_PATH"]):
                if (time.time() - os.path.getmtime(CONFIG["LATEST_IMAGE_PATH"])) < 600: image_fresh = True

            ai_is_safe, condition, confidence = False, "Waiting...", 0.0
            final_ai_safe = False

            if image_fresh:
                img = cv2.imread(CONFIG["LATEST_IMAGE_PATH"])
                if img is not None:
                    h, w = img.shape[:2]
                    cw, ch = int(CONFIG.get("INITIAL_CROP_SIZE_W", 1300)), int(CONFIG.get("INITIAL_CROP_SIZE_H", 1300))
                    if h >= ch and w >= cw:
                        sx, sy = (w - cw) // 2, (h - ch) // 2
                        img = img[sy:sy + ch, sx:sx + cw]
                    
                    # AI Processing
                    resized = cv2.resize(img, (224, 224))
                    data = np.expand_dims((resized.astype(np.float32) / 255.0), axis=0)
                    INTERPRETER.set_tensor(INPUT_DETAILS[0]['index'], data)
                    INTERPRETER.invoke()
                    out = INTERPRETER.get_tensor(OUTPUT_DETAILS[0]['index'])[0]
                    idx = np.argmax(out)
                    condition, confidence = CLASS_NAMES[idx], float(out[idx])
                    
                    safe_list = [s.strip() for s in CONFIG.get("SAFE_CONDITIONS", "").split(',')]
                    ai_is_safe = condition in safe_list
                
                app_instance.after(0, app_instance.update_image_display_thread_safe, Image.open(CONFIG["LATEST_IMAGE_PATH"]))
                
                now = datetime.now()
                is_ai_rain = "rain" in condition.lower()
                
                if ai_is_safe:
                    LAST_SAFE_TIME, IS_IN_GRACE_PERIOD, final_ai_safe = now, False, True
                elif is_ai_rain or LAST_SAFE_TIME is None:
                    final_ai_safe, IS_IN_GRACE_PERIOD = False, False
                else:
                    elapsed = (now - LAST_SAFE_TIME).total_seconds() / 60.0
                    grace_limit = float(CONFIG.get("CLOUD_GRACE_MINS", 5))
                    if elapsed < grace_limit:
                        final_ai_safe, IS_IN_GRACE_PERIOD = True, True
                        condition = f"{condition} ({max(0, grace_limit-elapsed):.1f}m left)"
                    else: final_ai_safe, IS_IN_GRACE_PERIOD = False, False

                station_safe = True
                if CONFIG.get("USE_WEATHER_STATION"):
                    station_safe = not (WEATHER_DATA["wind_speed"] > float(CONFIG["WIND_LIMIT"]) or WEATHER_DATA["is_raining"])
                
                IS_REALLY_SAFE_GLOBAL = final_ai_safe and station_safe
                CURRENT_STATUS = "SAFE" if IS_REALLY_SAFE_GLOBAL else "UNSAFE"
                CURRENT_CONDITION, CURRENT_CONFIDENCE = condition, confidence
            else:
                CURRENT_CONDITION = "IMAGE STALE"
                IS_REALLY_SAFE_GLOBAL = False
                CURRENT_STATUS = "UNSAFE"

            with open(CONFIG["ASCOM_FILE_PATH"], 'w') as f:
                f.write(f"IsSafe={IS_REALLY_SAFE_GLOBAL}\nCondition={CURRENT_CONDITION}\nConfidence={CURRENT_CONFIDENCE:.2f}\n")

            app_instance.after(0, app_instance.trigger_gui_refresh)
        except Exception: pass
        time.sleep(CONFIG["ASCOM_MONITOR_DELAY"])

# --- GUI ---

class AllSkyMonitorApp(tk.Tk):
    def __init__(self):
        load_config()
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.title("Observatory Guardian - AI Hybrid")
        self.geometry("1600x1000")
        self.configure(bg="#121212")
        self.image_display_width, self.image_display_height = 850, 550
        self.create_ui()
        threading.Thread(target=monitor_loop, args=(self,), daemon=True).start()
        if CONFIG.get("USE_WEB_SERVER"):
            threading.Thread(target=lambda: ThreadingHTTPServer(('0.0.0.0', int(CONFIG["WEB_SERVER_PORT"])), AlpacaDashboardHandler).serve_forever(), daemon=True).start()
            threading.Thread(target=run_alpaca_discovery_responder, daemon=True).start()
        if CONFIG.get("USE_WEATHER_STATION"):
            threading.Thread(target=lambda: ThreadingHTTPServer(('0.0.0.0', int(CONFIG["ECOWITT_PORT"])), EcowittHandler).serve_forever(), daemon=True).start()
        if CONFIG["FETCH_METHOD"] == "MQTT":
            threading.Thread(target=run_mqtt_listener, daemon=True).start()

    def create_ui(self):
        main = ttk.Frame(self, padding=10); main.pack(fill='both', expand=True)
        side = ttk.Frame(main, width=320); side.pack(side='left', fill='y', padx=(0, 15)); side.pack_propagate(False)
        
        ttk.Label(side, text="Safety Status", font=('Segoe UI', 16, 'bold')).pack(pady=10)
        self.canvas_status = tk.Canvas(side, width=110, height=110, bg="#121212", highlightthickness=0); self.canvas_status.pack()
        self.indicator = self.canvas_status.create_oval(10, 10, 100, 100, fill="gray")
        self.stat_lbl = ttk.Label(side, text="LOADING...", font=('Segoe UI', 18, 'bold')); self.stat_lbl.pack()
        self.ai_lbl = ttk.Label(side, text="AI: Initializing...", font=('Segoe UI', 10)); self.ai_lbl.pack()
        self.conf_lbl = ttk.Label(side, text="Conf: 0%", font=('Segoe UI', 9)); self.conf_lbl.pack()
        
        ttk.Separator(side, orient='horizontal').pack(fill='x', pady=15)
        self.mqtt_status_lbl = ttk.Label(side, text="MQTT: Disconnected", font=('Segoe UI', 9)); self.mqtt_status_lbl.pack(anchor='w')
        
        ttk.Separator(side, orient='horizontal').pack(fill='x', pady=10)
        self.compass = CompassWidget(side); self.compass.pack(pady=10)
        
        m_frame = ttk.Frame(side); m_frame.pack(fill='x', padx=10)
        self.metrics_lbls = {}
        for k in ['Temp', 'Humidity', 'Pressure', 'Wind', 'Gust', 'Rain', 'Solar', 'UV']:
            f = ttk.Frame(m_frame); f.pack(fill='x', pady=1)
            ttk.Label(f, text=k + ":").pack(side='left')
            l = ttk.Label(f, text="--", font=('Arial', 9, 'bold')); l.pack(side='right')
            self.metrics_lbls[k] = l
        
        ttk.Button(side, text="⚙ Configure Settings", command=lambda: SettingsWindow(self, CONFIG)).pack(side='bottom', pady=20)

        right = ttk.Frame(main); right.pack(side='right', fill='both', expand=True)
        self.img_lbl = ttk.Label(right, text="Awaiting Image..."); self.img_lbl.pack(pady=5)
        
        g_frame = ttk.Frame(right); g_frame.pack(fill='both', expand=True, pady=10)
        self.graphs = {}
        g_defs = [
            ("Temp (°C)", "#FF5722", 1, 0, 0), ("Humidity (%)", "#2196F3", 2, 0, 1), ("Pressure (hPa)", "#9C27B0", 3, 0, 2),
            ("Wind (km/h)", "#4CAF50", 4, 1, 0), ("Rain (in/hr)", "#03A9F4", 6, 1, 1), ("Solar (W/m²)", "#FFC107", 7, 1, 2)
        ]
        for title, col, idx, r, c in g_defs:
            g = MiniGraph(g_frame, title, col, height=135)
            g.grid(row=r, column=c, sticky='ew', padx=2, pady=2)
            self.graphs[idx] = g
            g_frame.columnconfigure(c, weight=1)

    def trigger_gui_refresh(self):
        if CURRENT_STATUS == "AWAITING": color = "#888888" 
        elif CURRENT_STATUS == "SAFE": color = "#FF9800" if IS_IN_GRACE_PERIOD else "#4CAF50"
        else: color = "#F44336"
        self.canvas_status.itemconfig(self.indicator, fill=color)
        self.stat_lbl.config(text=CURRENT_STATUS, foreground=color)
        self.ai_lbl.config(text=f"AI: {CURRENT_CONDITION}")
        self.conf_lbl.config(text=f"Confidence: {CURRENT_CONFIDENCE*100:.1f}%")
        
        self.metrics_lbls['Temp'].config(text=f"{WEATHER_DATA['temp']} °C")
        self.metrics_lbls['Humidity'].config(text=f"{WEATHER_DATA['humidity']} %")
        self.metrics_lbls['Pressure'].config(text=f"{WEATHER_DATA['pressure']} hPa")
        self.metrics_lbls['Wind'].config(text=f"{WEATHER_DATA['wind_speed']:.1f} km/h")
        self.metrics_lbls['Gust'].config(text=f"{WEATHER_DATA['wind_gust']:.1f} km/h")
        self.metrics_lbls['Rain'].config(text=f"{WEATHER_DATA['rain_rate']} in/hr")
        self.metrics_lbls['Solar'].config(text=f"{WEATHER_DATA['solar']} W/m²")
        self.metrics_lbls['UV'].config(text=f"{WEATHER_DATA['uv']}")
        
        self.mqtt_status_lbl.config(text=f"MQTT: {MQTT_CONN_STATUS}")
        self.compass.update_dir(WEATHER_DATA["wind_dir"])
        for idx, g in self.graphs.items(): g.update_data(idx)

    def update_image_display_thread_safe(self, pil_img):
        dw, dh = self.image_display_width, self.image_display_height
        pil_img.thumbnail((dw, dh), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img); self.img_lbl.config(image=self.tk_img, text="")

    def on_closing(self): self.destroy()

class SettingsWindow(tk.Toplevel):
    def __init__(self, master, current_config):
        super().__init__(master)
        self.title("Configuration"); self.config = current_config.copy(); self.grab_set()
        c = ttk.Frame(self, padding=10); c.pack(fill='both', expand=True); self.entries = {}
        
        nb = ttk.Notebook(c); nb.pack(fill='both', expand=True)
        
        # --- TAB 1: GENERAL ---
        t1 = ttk.Frame(nb, padding=10); nb.add(t1, text='General')
        self.add_f(t1, "Local Image Cache", "LATEST_IMAGE_PATH", 0, "save")
        self.add_f(t1, "ASCOM Status File", "ASCOM_FILE_PATH", 1, "save")
        self.add_f(t1, "Weather Log File", "LOG_FILE_PATH", 2, "save")
        self.add_f(t1, "Monitor Delay (s)", "ASCOM_MONITOR_DELAY", 3)
        
        sep = ttk.LabelFrame(t1, text="Web Server & Alpaca", padding=5); sep.grid(row=4, column=0, columnspan=3, sticky='ew', pady=10)
        use_web = tk.BooleanVar(value=self.config.get("USE_WEB_SERVER", True))
        ttk.Checkbutton(sep, text="Enable Server", variable=use_web).grid(row=0, column=0, sticky='w')
        self.entries["USE_WEB_SERVER"] = use_web
        self.add_f(sep, "Port", "WEB_SERVER_PORT", 1)
        self.add_f(sep, "Device #", "ALPACA_DEVICE_NUMBER", 2)
        
        # --- TAB 2: AI LOGIC ---
        t2 = ttk.Frame(nb, padding=10); nb.add(t2, text='AI Logic')
        self.add_f(t2, "Model (.tflite)", "MODEL_PATH", 0, "open")
        self.add_f(t2, "Labels (.txt)", "LABELS_PATH", 1, "open")
        self.add_f(t2, "Safe AI Labels", "SAFE_CONDITIONS", 2)
        self.add_f(t2, "Cloud Grace (m)", "CLOUD_GRACE_MINS", 3)
        
        crop_f = ttk.LabelFrame(t2, text="Image Cropping", padding=5); crop_f.grid(row=4, column=0, columnspan=3, sticky='ew', pady=10)
        self.add_f(crop_f, "Crop Width", "INITIAL_CROP_SIZE_W", 0)
        self.add_f(crop_f, "Crop Height", "INITIAL_CROP_SIZE_H", 1)

        # --- TAB 3: CONNECTION ---
        t3 = ttk.Frame(nb, padding=10); nb.add(t3, text='Connection')
        self.method_var = tk.StringVar(value=self.config.get("FETCH_METHOD", "MQTT"))
        f_m = ttk.LabelFrame(t3, text="Fetch Method", padding=5); f_m.grid(row=0, column=0, columnspan=3, sticky='ew', pady=5)
        ttk.Radiobutton(f_m, text="MQTT", variable=self.method_var, value="MQTT").pack(side='left', padx=10)
        ttk.Radiobutton(f_m, text="SFTP", variable=self.method_var, value="SFTP").pack(side='left', padx=10)
        ttk.Radiobutton(f_m, text="IP Cam", variable=self.method_var, value="IP_CAM").pack(side='left', padx=10)
        
        # MQTT
        f_mq = ttk.LabelFrame(t3, text="MQTT Broker", padding=5); f_mq.grid(row=1, column=0, sticky='nsew', padx=2)
        self.add_f(f_mq, "IP/Host", "MQTT_BROKER", 0)
        self.add_f(f_mq, "Port", "MQTT_PORT", 1)
        self.add_f(f_mq, "Topic", "MQTT_TOPIC", 2)
        self.add_f(f_mq, "User", "MQTT_USER", 3)
        self.add_f(f_mq, "Pass", "MQTT_PASS", 4, show="*")
        
        # SFTP
        f_sf = ttk.LabelFrame(t3, text="SFTP Remote", padding=5); f_sf.grid(row=1, column=1, sticky='nsew', padx=2)
        self.add_f(f_sf, "Host IP", "ALLSKY_HOST", 0)
        self.add_f(f_sf, "User", "ALLSKY_USER", 1)
        self.add_f(f_sf, "Pass", "ALLSKY_PASS", 2, show="*")
        self.add_f(f_sf, "Path", "REMOTE_IMAGE_PATH", 3)
        self.add_f(f_sf, "Max Retries", "SFTP_MAX_RETRIES", 4)
        self.add_f(f_sf, "Retry Delay", "SFTP_RETRY_DELAY", 5)
        
        # IP Cam
        f_ip = ttk.LabelFrame(t3, text="IP Camera Stream", padding=5); f_ip.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        self.add_f(f_ip, "Stream URL", "IP_CAM_URL", 0)

        # --- TAB 4: WEATHER ---
        t4 = ttk.Frame(nb, padding=10); nb.add(t4, text='Weather')
        
        ecowitt_f = ttk.LabelFrame(t4, text="Ecowitt Station", padding=5); ecowitt_f.pack(fill='x', pady=5)
        use_ws = tk.BooleanVar(value=self.config.get("USE_WEATHER_STATION", True))
        ttk.Checkbutton(ecowitt_f, text="Enable Listener", variable=use_ws).grid(row=0, column=0, sticky='w')
        self.entries["USE_WEATHER_STATION"] = use_ws
        self.add_f(ecowitt_f, "Port", "ECOWITT_PORT", 1)
        self.add_f(ecowitt_f, "Wind Limit", "WIND_LIMIT", 2)
        self.add_f(ecowitt_f, "Humidity Limit", "HUMIDITY_LIMIT", 3)
        
        proxy_f = ttk.LabelFrame(t4, text="Data Forwarding (Proxy)", padding=5); proxy_f.pack(fill='x', pady=5)
        use_fw = tk.BooleanVar(value=self.config.get("FORWARD_DATA", False))
        ttk.Checkbutton(proxy_f, text="Enable Forwarding", variable=use_fw).grid(row=0, column=0, sticky='w')
        self.entries["FORWARD_DATA"] = use_fw
        self.add_f(proxy_f, "Forward URL", "FORWARD_URL", 1)

        ttk.Button(c, text="✅ Save & Apply", command=self.save).pack(pady=10)

    def add_f(self, p, lbl, key, row, mode=None, show=None):
        ttk.Label(p, text=lbl).grid(row=row, column=0, sticky='w', padx=2, pady=2)
        e = ttk.Entry(p, width=25, show=show); e.insert(0, str(self.config.get(key, ""))); e.grid(row=row, column=1, padx=2, pady=2)
        self.entries[key] = e
        if mode: ttk.Button(p, text="...", width=3, command=lambda: self.on_b(key, mode)).grid(row=row, column=2)

    def on_b(self, key, mode):
        p = filedialog.asksaveasfilename() if mode == "save" else filedialog.askopenfilename()
        if p: self.entries[key].delete(0, tk.END); self.entries[key].insert(0, p)

    def save(self):
        new_conf = {}
        for k, e in self.entries.items():
            if isinstance(e, tk.BooleanVar): 
                new_conf[k] = e.get()
            else:
                v = e.get()
                # Determine type for casting
                numeric_fields = [
                    "WIND_LIMIT", "HUMIDITY_LIMIT", "CLOUD_GRACE_MINS", 
                    "WEB_SERVER_PORT", "ALPACA_DEVICE_NUMBER", "ASCOM_MONITOR_DELAY", 
                    "ECOWITT_PORT", "MQTT_PORT", "INITIAL_CROP_SIZE_W", 
                    "INITIAL_CROP_SIZE_H", "SFTP_MAX_RETRIES", "SFTP_RETRY_DELAY"
                ]
                if k in numeric_fields:
                    try: new_conf[k] = float(v) if "." in str(v) else int(v)
                    except: new_conf[k] = DEFAULT_CONFIG.get(k, 0)
                else:
                    new_conf[k] = v
        
        new_conf["FETCH_METHOD"] = self.method_var.get()
        if save_config(new_conf): 
            messagebox.showinfo("Success", "All Settings Saved!"); self.destroy()

if __name__ == '__main__':
    AllSkyMonitorApp().mainloop()