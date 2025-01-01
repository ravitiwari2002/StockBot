from flask import Flask, render_template, redirect
import subprocess
import sys
import logging
import os
import socket

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

streamlit_processes = {}

# Configurations for Streamlit apps
STREAMLIT_APPS = {
    'chatbot': {'file': 'chatbot.py', 'port': 8501},
    'forecast': {'file': 'forecast.py', 'port': 8502}
}

def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_streamlit_app(app_config):
    try:
        script_path = app_config['file']

        # Check if the script file exists
        if not os.path.exists(script_path):
            logging.error(f"Script file does not exist: {script_path}")
            return None

        # Check if the port is already in use
        if is_port_in_use(app_config['port']):
            logging.warning(f"Port {app_config['port']} is already in use. Skipping startup for {script_path}.")
            return None

        command = [
            sys.executable,
            "-m", "streamlit", "run", script_path,
            f"--server.port={app_config['port']}",
            "--server.headless=true"
        ]
        process = subprocess.Popen(command)
        logging.info(f"Started Streamlit app: {script_path} on port {app_config['port']}")
        return process
    except Exception as e:
        logging.error(f"Error starting Streamlit app {script_path}: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run/<app_key>')
def run_app(app_key):
    app_config = STREAMLIT_APPS.get(app_key)
    if not app_config:
        return f"App '{app_key}' not found.", 404
    return redirect(f"http://localhost:{app_config['port']}")

if __name__ == '__main__':
    if os.getenv('WERKZEUG_RUN_MAIN') == 'true':  # Only start subprocesses in the main instance
        for app_key, config in STREAMLIT_APPS.items():
            streamlit_processes[app_key] = start_streamlit_app(config)

    app.run(debug=True)