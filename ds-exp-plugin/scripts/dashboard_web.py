#!/usr/bin/env python3
"""
ML Experiment Dashboard - Web UI
Launches Gradio interface for viewing experiments
"""
import gradio as gr
import subprocess
import json
import os
from datetime import datetime

def get_dashboard_data():
    """Get dashboard data as structured dict"""
    data = {
        'gpus': [],
        'processes': [],
        'experiments': []
    }

    # Get GPU info
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        data['gpus'].append({
                            'id': parts[0],
                            'name': parts[1],
                            'util': parts[2],
                            'mem_used': parts[3],
                            'mem_total': parts[4],
                            'temp': parts[5]
                        })
    except:
        pass

    # Get experiments
    import glob

    # Get current session ID if available
    current_session = None
    if os.path.exists('.claude_session'):
        with open('.claude_session', 'r') as f:
            current_session = f.read().strip()

    # Find experiment files
    if current_session:
        exp_files = glob.glob(f"experiments/session-{current_session}/exp_*.json")
    else:
        exp_files = glob.glob("experiments/**/exp_*.json", recursive=True)

    for exp_file in exp_files:
        try:
            with open(exp_file, 'r') as f:
                exp_data = json.load(f)

                # Calculate time ago
                start_time_str = exp_data.get('start_time', '')
                time_ago = ''
                if start_time_str:
                    try:
                        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
                        now = datetime.now(start_time.tzinfo)
                        delta = now - start_time

                        if delta.days > 0:
                            time_ago = f"{delta.days}d ago"
                        elif delta.seconds >= 3600:
                            hours = delta.seconds // 3600
                            time_ago = f"{hours}h ago"
                        elif delta.seconds >= 60:
                            minutes = delta.seconds // 60
                            time_ago = f"{minutes}m ago"
                        else:
                            time_ago = f"{delta.seconds}s ago"
                    except:
                        pass

                data['experiments'].append({
                    'name': exp_data.get('name', 'unknown'),
                    'status': exp_data.get('status', 'unknown'),
                    'time_ago': time_ago,
                    'params': exp_data.get('params', {}),
                    'metrics': exp_data.get('metrics', {}),
                    'notes': exp_data.get('notes', []),
                    'gpu': exp_data.get('gpu'),
                    'file': exp_file
                })
        except:
            pass

    # Sort by status (running first) then by time
    data['experiments'].sort(key=lambda x: (x['status'] != 'running', x['time_ago']))

    return data

def render_dashboard():
    """Render dashboard as HTML"""
    data = get_dashboard_data()

    html = "<style>"
    html += "body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; }"
    html += ".gpu-idle { color: #4ec9b0; }"
    html += ".gpu-busy { color: #ce9178; }"
    html += ".gpu-maxed { color: #f48771; }"
    html += ".status-running { color: #dcdcaa; }"
    html += ".status-completed { color: #4ec9b0; }"
    html += ".status-failed { color: #f48771; }"
    html += "table { border-collapse: collapse; width: 100%; margin: 20px 0; }"
    html += "th, td { padding: 8px; text-align: left; border-bottom: 1px solid #3e3e3e; }"
    html += "th { background: #2d2d30; }"
    html += ".note { color: #858585; font-style: italic; margin: 4px 0; }"
    html += "</style>"

    html += "<h1>🔥 ML Experiment Dashboard</h1>"

    # GPU Section
    html += "<h2>GPUs</h2>"
    if data['gpus']:
        html += "<table>"
        html += "<tr><th>ID</th><th>Name</th><th>Util</th><th>Memory</th><th>Temp</th></tr>"
        for gpu in data['gpus']:
            util = int(gpu['util'])
            if util < 10:
                css_class = "gpu-idle"
            elif util < 70:
                css_class = "gpu-busy"
            else:
                css_class = "gpu-maxed"

            html += f"<tr class='{css_class}'>"
            html += f"<td>{gpu['id']}</td>"
            html += f"<td>{gpu['name']}</td>"
            html += f"<td>{gpu['util']}%</td>"
            html += f"<td>{gpu['mem_used']}/{gpu['mem_total']} MB</td>"
            html += f"<td>{gpu['temp']}°C</td>"
            html += "</tr>"
        html += "</table>"
    else:
        html += "<p>No GPUs detected</p>"

    # Experiments Section
    html += "<h2>Experiments</h2>"
    if data['experiments']:
        html += "<table>"
        html += "<tr><th>Name</th><th>Status</th><th>Time</th><th>GPU</th><th>Metrics</th><th>Notes</th></tr>"
        for exp in data['experiments']:
            status_class = f"status-{exp['status']}"

            # Format metrics
            metrics_str = ", ".join([f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                                     for k, v in list(exp['metrics'].items())[:3]])

            # Format notes (last 2)
            notes_html = ""
            for note in exp['notes'][-2:]:
                time_str = note.get('time', '')[:16]
                text = note.get('text', '')
                notes_html += f"<div class='note'>[{time_str}] {text}</div>"

            html += f"<tr class='{status_class}'>"
            html += f"<td>{exp['name']}</td>"
            html += f"<td>{exp['status']}</td>"
            html += f"<td>{exp['time_ago']}</td>"
            html += f"<td>GPU {exp['gpu']}</td>"
            html += f"<td>{metrics_str or 'N/A'}</td>"
            html += f"<td>{notes_html or '-'}</td>"
            html += "</tr>"
        html += "</table>"
    else:
        html += "<p>No experiments tracked yet</p>"

    return html

def launch_dashboard():
    """Launch Gradio dashboard"""
    with gr.Blocks(title="DS-Exp Dashboard") as demo:
        gr.Markdown("# 🔥 ML Experiment Dashboard")

        html_output = gr.HTML()
        refresh_btn = gr.Button("🔄 Refresh", variant="primary")

        # Initial load
        demo.load(fn=render_dashboard, outputs=html_output)

        # Refresh on button click
        refresh_btn.click(fn=render_dashboard, outputs=html_output)

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML Experiment Dashboard Web UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Detect SageMaker Studio environment
    studio_url = None
    try:
        import json
        with open('/opt/ml/metadata/resource-metadata.json', 'r') as f:
            metadata = json.load(f)
            domain_id = metadata.get('DomainId', '')
            region = os.environ.get('AWS_REGION', os.environ.get('REGION_NAME', 'us-east-1'))

            if domain_id:
                # Extract short domain ID (everything after 'd-')
                if domain_id.startswith('d-'):
                    short_domain_id = domain_id[2:]
                    # SageMaker Studio URL pattern
                    studio_url = f"https://{short_domain_id}.studio.{region}.sagemaker.aws/jupyterlab/default/proxy/{args.port}/"
    except:
        pass

    print("🚀 Launching dashboard web UI...")
    if studio_url:
        print(f"📊 Dashboard URL (SageMaker Studio):")
        print(f"   {studio_url}")
    else:
        print(f"📊 Dashboard will be available at: http://localhost:{args.port}")

    demo = launch_dashboard()
    demo.launch(
        server_port=args.port,
        share=args.share,
        quiet=False,
        theme=gr.themes.Soft(),
        inbrowser=False  # Don't auto-open in SageMaker (won't work)
    )
