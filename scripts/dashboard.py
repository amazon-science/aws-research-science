#!/usr/bin/env python3
"""
ML Experiment Dashboard - Compact view of experiments, GPUs, results
Usage: python dashboard.py [--refresh 5]
"""
import subprocess
import json
import yaml
import time
import argparse
import os
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.rule import Rule

import shutil
_tw = shutil.get_terminal_size(fallback=(100, 40)).columns
console = Console(width=_tw, highlight=False, color_system="256")

def get_gpu_status():
    """Get GPU utilization and memory"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, name, util_gpu, mem_used, mem_total, temp = line.split(',')
                gpus.append({
                    'id': int(idx.strip()),
                    'name': name.strip(),
                    'util_gpu': int(util_gpu.strip()),
                    'mem_used': int(mem_used.strip()),
                    'mem_total': int(mem_total.strip()),
                    'temp': int(temp.strip())
                })
        return gpus
    except:
        return []

def get_running_processes():
    """Get running training processes"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=2
        )
        processes = []
        for line in result.stdout.split('\n'):
            if 'python' in line.lower() and any(kw in line.lower() for kw in ['train', 'experiment', 'rl', 'sft']):
                if 'grep' not in line:
                    parts = line.split()
                    if len(parts) > 10:
                        processes.append({
                            'pid': parts[1],
                            'cpu': parts[2],
                            'mem': parts[3],
                            'cmd': ' '.join(parts[10:])
                        })
        return processes
    except:
        return []

def get_json_experiments(session_filter=True):
    """Get all experiments from JSON files"""
    try:
        import json
        import glob
        from datetime import datetime
        import os

        # Get current session ID if filtering
        current_session = None
        if session_filter and os.path.exists('.claude_session'):
            with open('.claude_session', 'r') as f:
                current_session = f.read().strip()

        # Find experiment files
        if session_filter and current_session:
            exp_files = glob.glob(f"experiments/session-{current_session}/exp_*.json")
        else:
            exp_files = glob.glob("experiments/**/exp_*.json", recursive=True)

        experiments = []

        for exp_file in exp_files:
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)

                # Extract primary metric (first metric with highest value)
                metrics = data.get('metrics', {})
                primary_metric = 'N/A'
                primary_value = 0

                if metrics:
                    # Find metric with highest value
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)) and metric_value > primary_value:
                            primary_metric = metric_name
                            primary_value = metric_value

                # Calculate time since start
                start_time_str = data.get('start_time', '')
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
                        time_ago = ''

                experiments.append({
                    'branch': data.get('name', 'unknown'),
                    'status': data.get('status', 'unknown'),
                    'value': primary_value,
                    'metric': primary_metric,
                    'time_ago': time_ago,
                    'file': exp_file
                })
            except:
                pass

        return experiments
    except:
        return []

def get_results_summary():
    """Get summary from JSON experiment files"""
    try:
        import json
        import glob

        exp_files = glob.glob("experiments/**/exp_*.json", recursive=True) + \
                    glob.glob("experiments/exp_*.json")
        if not exp_files:
            return None

        completed = 0
        failed = 0
        running = 0

        for exp_file in exp_files:
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                    status = data.get('status', 'unknown')
                    if status == 'completed':
                        completed += 1
                    elif status == 'failed':
                        failed += 1
                    elif status == 'running':
                        running += 1
            except:
                pass

        return {
            'total': completed + failed + running,
            'completed': completed,
            'failed': failed,
            'running': running
        }
    except:
        return None

def get_disk_usage():
    """Get disk usage"""
    try:
        result = subprocess.run(
            ["df", "-h", "."],
            capture_output=True, text=True, timeout=2
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            return {
                'total': parts[1],
                'used': parts[2],
                'available': parts[3],
                'percent': int(parts[4].replace('%', ''))
            }
    except:
        return None

def get_queue_status():
    """Get queue status"""
    try:
        import json
        queue_file = "experiments/queue.json"
        if os.path.exists(queue_file):
            with open(queue_file, 'r') as f:
                queue = json.load(f)
                return {
                    'queued': queue.get('queued', []),
                    'running': queue.get('running', [])
                }
        return {'queued': [], 'running': []}
    except:
        return {'queued': [], 'running': []}

def util_bar(pct, width=10):
    """Render a mini utilization bar"""
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    if pct > 90:
        return f"[red]{bar}[/red]"
    elif pct > 60:
        return f"[yellow]{bar}[/yellow]"
    else:
        return f"[green]{bar}[/green]"


def generate_compact_dashboard():
    """Generate compact single-screen dashboard"""
    gpus = get_gpu_status()
    session_filter = globals().get('SESSION_FILTER', True)
    experiments = get_json_experiments(session_filter=session_filter)
    processes = get_running_processes()
    results = get_results_summary()
    disk = get_disk_usage()
    queue = get_queue_status()

    # ── Header ──────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%H:%M:%S")
    header = Text()
    header.append("🔥 ML Dashboard", style="bold cyan")
    header.append(f"  {now}", style="dim")
    if results:
        header.append(f"   📊 {results['total']} exp ", style="bold white")
        header.append(f"✅{results['completed']} ", style="green")
        header.append(f"❌{results['failed']} ", style="red")
        header.append(f"🔄{results.get('running',0)}", style="yellow")
    if disk:
        pct = disk['percent']
        col = "red" if pct > 90 else "yellow" if pct > 75 else "dim"
        header.append(f"   💾 {disk['used']}/{disk['total']}", style=col)
    if queue:
        r, q = len(queue['running']), len(queue['queued'])
        if r or q:
            header.append(f"   ⚡{r} running  ⏳{q} queued", style="cyan")
    console.print(header)
    console.print()

    # ── Queue ────────────────────────────────────────────────────────────────
    if queue and (queue['running'] or queue['queued']):
        q_table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), header_style="bold",
                        show_edge=False)
        q_table.add_column("Job", min_width=26)
        q_table.add_column("GPU", justify="center", min_width=5)
        q_table.add_column("Status", min_width=10)
        q_table.add_column("Started", style="dim")

        for job in queue['running']:
            started = job.get('started_at', '')[:16].replace('T', ' ')
            q_table.add_row(
                job.get('name', '')[:28],
                f"[cyan]{job.get('gpu')}[/cyan]",
                "[green]● running[/green]",
                started,
            )
        for job in queue['queued'][:6]:
            retry = job.get('retry_count', 0)
            retry_str = f" ×{retry}" if retry else ""
            q_table.add_row(
                job.get('name', '')[:28],
                "[dim]—[/dim]",
                f"[yellow]◌ waiting{retry_str}[/yellow]",
                "",
            )

        console.print(Rule("Queue", style="dim", align="left"))
        console.print(q_table)
        console.print()

    # ── GPUs ─────────────────────────────────────────────────────────────────
    if gpus:
        g_table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), header_style="bold",
                        show_edge=False)
        g_table.add_column("#", style="cyan", justify="right", min_width=2)
        g_table.add_column("Name", min_width=10)
        g_table.add_column("Util", min_width=14)
        g_table.add_column("Mem", justify="right", min_width=10)
        g_table.add_column("Temp", justify="right", min_width=5)

        for gpu in gpus:
            util = gpu['util_gpu']
            mem_pct = int(gpu['mem_used'] / gpu['mem_total'] * 100)
            mem_col = "red" if mem_pct > 90 else "yellow" if mem_pct > 70 else "green"
            temp_col = "red" if gpu['temp'] > 80 else "yellow" if gpu['temp'] > 70 else "green"
            g_table.add_row(
                str(gpu['id']),
                gpu['name'][:12],
                f"{util_bar(util)}  [{('red' if util>90 else 'yellow' if util>60 else 'green')}]{util:3d}%[/]",
                f"[{mem_col}]{gpu['mem_used']//1024}/{gpu['mem_total']//1024}GB[/{mem_col}]",
                f"[{temp_col}]{gpu['temp']}°C[/{temp_col}]",
            )

        console.print(Rule("GPUs", style="dim", align="left"))
        console.print(g_table)
        console.print()

    # ── Processes ─────────────────────────────────────────────────────────────
    if processes:
        p_table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), header_style="bold",
                        show_edge=False)
        p_table.add_column("PID", style="cyan", min_width=7)
        p_table.add_column("CPU", justify="right", min_width=5)
        p_table.add_column("Mem", justify="right", min_width=5)
        p_table.add_column("Command", )

        for proc in processes[:3]:
            cpu = float(proc['cpu'])
            cpu_col = "red" if cpu > 90 else "yellow" if cpu > 50 else "green"
            cmd = proc['cmd'][:65] + "…" if len(proc['cmd']) > 65 else proc['cmd']
            p_table.add_row(
                proc['pid'],
                f"[{cpu_col}]{cpu:.0f}%[/{cpu_col}]",
                f"{float(proc['mem']):.1f}%",
                cmd,
            )

        console.print(Rule("Processes", style="dim", align="left"))
        console.print(p_table)
        console.print()

    # ── Experiments ───────────────────────────────────────────────────────────
    if experiments:
        experiments = sorted(experiments, key=lambda x: x.get('value') or 0, reverse=True)
        e_table = Table(box=box.SIMPLE_HEAD, padding=(0, 1), header_style="bold",
                        show_edge=False)
        e_table.add_column("Experiment", min_width=26)
        e_table.add_column("", min_width=2)
        e_table.add_column("Age", style="dim", min_width=6)
        e_table.add_column("Best Metric", min_width=20)

        for exp in experiments[:5]:
            status = exp.get('status', '')
            icon = {"completed": "[green]✅[/green]", "running": "[yellow]🔄[/yellow]",
                    "failed": "[red]❌[/red]"}.get(status, "[dim]·[/dim]")

            value = exp.get('value', 0)
            metric = exp.get('metric', '')
            if value and metric != 'N/A':
                col = "green" if value > 0.5 else "yellow" if value > 0.1 else "white"
                m_str = f"[{col}]{metric} = {value:.4f}[/{col}]"
            else:
                m_str = "[dim]—[/dim]"

            e_table.add_row(
                exp['branch'][:28],
                icon,
                exp.get('time_ago', '')[:7],
                m_str,
            )

        console.print(Rule("Experiments", style="dim", align="left"))
        console.print(e_table)
    else:
        console.print(Rule("Experiments", style="dim", align="left"))
        console.print("[dim]  No experiments tracked yet[/dim]")

def main():
    parser = argparse.ArgumentParser(description="ML Experiment Dashboard")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--all", action="store_true", help="Show all sessions (not just current)")
    args = parser.parse_args()

    # Set global session filter
    global SESSION_FILTER
    SESSION_FILTER = not args.all

    if args.once:
        # Single render
        generate_compact_dashboard()
    else:
        # Live updating dashboard
        try:
            while True:
                console.clear()
                generate_compact_dashboard()
                console.print(f"\n[dim]Press Ctrl+C to exit | Refreshing every {args.refresh}s...[/dim]")
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped[/yellow]")

if __name__ == "__main__":
    main()
