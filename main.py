from flask import Flask, render_template, redirect
import subprocess
import sys
import atexit
import signal
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

streamlit_processes = {}

# Configurations for Streamlit apps
STREAMLIT_APPS = {
    'chatbot': {'name': os.getenv('CHATBOT_NAME', '💬 Stock Chatbot Assistant'), 'port': int(os.getenv('CHATBOT_PORT', 8501))},
    'forecast': {'name': os.getenv('FORECAST_NAME', 'Stock Prophet'), 'port': int(os.getenv('FORECAST_PORT', 8502))}
}

def start_streamlit_app(app_config):
    try:
        command = [
            sys.executable,
            "-m", "streamlit", "run", f"{app_config['name']}.py",
            f"--server.port={app_config['port']}",
            "--server.headless=true"
        ]
        process = subprocess.Popen(command)
        logging.info(f"Started {app_config['name']} on port {app_config['port']}")
        return process
    except Exception as e:
        logging.error(f"Error starting {app_config['name']}: {e}")
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

# Cleanup subprocesses on exit
def cleanup():
    logging.info("Cleaning up Streamlit processes...")
    for process in streamlit_processes.values():
        process.terminate()

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: cleanup())

if __name__ == '__main__':
    for app_key, config in STREAMLIT_APPS.items():
        streamlit_processes[app_key] = start_streamlit_app(config)

    app.run(debug=True)
