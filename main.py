from flask import Flask, render_template, redirect
import subprocess
import sys

app = Flask(__name__)

streamlit_processes = {}

def start_streamlit_app(app_name, port):
    command = [
        sys.executable,
        "-m", "streamlit", "run", f"{app_name}.py",
        f"--server.port={port}",
        "--server.headless=true"  # Run in headless mode
    ]
    process = subprocess.Popen(command)
    return process

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_chatbot')
def run_chatbot():
    return redirect("http://localhost:8501")

@app.route('/run_forecast')
def run_forecast():
    return redirect("http://localhost:8502")

if __name__ == '__main__':
    # Start Streamlit apps
    streamlit_processes['chatbot'] = start_streamlit_app('chatbot', 8501)
    streamlit_processes['forecast'] = start_streamlit_app('forecast', 8502)

    app.run(debug=True)

    # Clean up subprocesses on exit
    for process in streamlit_processes.values():
        process.terminate()
