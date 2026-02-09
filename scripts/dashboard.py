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

console = Console()

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

        exp_files = glob.glob("experiments/exp_*.json")
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

def generate_compact_dashboard():
    """Generate compact single-screen dashboard"""
    # Get data
    gpus = get_gpu_status()
    session_filter = globals().get('SESSION_FILTER', True)
    experiments = get_json_experiments(session_filter=session_filter)
    processes = get_running_processes()
    results = get_results_summary()
    disk = get_disk_usage()
    queue = get_queue_status()

    # Header with summary
    header = Text()
    header.append("🔥 ML Experiment Dashboard", style="bold cyan")
    header.append(f"  |  {datetime.now().strftime('%H:%M:%S')}", style="dim")

    if results:
        header.append(f"  |  📊 {results['total']} exp ", style="bold")
        header.append(f"(✅{results['completed']} ", style="green")
        header.append(f"❌{results['failed']} ", style="red")
        header.append(f"🔄{results.get('running', 0)})", style="yellow")

    if disk:
        pct = disk['percent']
        style = "red" if pct > 90 else "yellow" if pct > 75 else "green"
        header.append(f"  |  💾 {disk['used']}/{disk['total']}", style=style)

    if queue:
        queued_count = len(queue['queued'])
        running_count = len(queue['running'])
        if queued_count > 0 or running_count > 0:
            header.append(f"  |  🔄 Queue: {running_count} running, {queued_count} waiting", style="yellow")

    console.print(header)
    console.print()

    # Queue Section
    if queue and (queue['queued'] or queue['running']):
        queue_table = Table(show_header=True, header_style="bold yellow", box=None, padding=0)
        queue_table.add_column("Job", style="white")
        queue_table.add_column("Status", no_wrap=True)
        queue_table.add_column("Info", style="dim")

        # Show running jobs
        for job in queue['running']:
            started = job.get('started_at', '')[:16]
            queue_table.add_row(
                job.get('name', 'unknown')[:30],
                f"[green]✓ GPU {job.get('gpu')}[/green]",
                f"Started {started}"
            )

        # Show queued jobs
        for job in queue['queued'][:5]:  # Show top 5
            retry = job.get('retry_count', 0)
            last_error = job.get('last_error', '')
            retry_str = f" (retry {retry}/3)" if retry > 0 else ""
            error_str = f" - {last_error[:30]}" if last_error else ""

            queue_table.add_row(
                job.get('name', 'unknown')[:30],
                f"[yellow]⏳ Waiting{retry_str}[/yellow]",
                f"Queued{error_str}"
            )

        console.print(queue_table)
        console.print()

    # GPU Table - Compact
    if gpus:
        gpu_table = Table(show_header=True, header_style="bold magenta", box=None, padding=0)
        gpu_table.add_column("GPU", style="cyan", no_wrap=True)
        gpu_table.add_column("Name", style="white")
        gpu_table.add_column("Util", justify="right", no_wrap=True)
        gpu_table.add_column("Memory", justify="right", no_wrap=True)
        gpu_table.add_column("Temp", justify="right", no_wrap=True)
        gpu_table.add_column("St", no_wrap=True)

        for gpu in gpus:
            util = gpu['util_gpu']
            mem_pct = int((gpu['mem_used'] / gpu['mem_total']) * 100)

            # Status
            if util < 10 and mem_pct < 50:
                status = "[green]✓[/green]"
            elif util < 50:
                status = "[yellow]~[/yellow]"
            else:
                status = "[red]![/red]"

            # Color coding
            util_str = f"{util}%"
            if util > 90:
                util_str = f"[red]{util_str}[/red]"
            elif util > 50:
                util_str = f"[yellow]{util_str}[/yellow]"
            else:
                util_str = f"[green]{util_str}[/green]"

            mem_str = f"{gpu['mem_used']//1024}GB/{gpu['mem_total']//1024}GB"
            if mem_pct > 90:
                mem_str = f"[red]{mem_str}[/red]"
            elif mem_pct > 70:
                mem_str = f"[yellow]{mem_str}[/yellow]"
            else:
                mem_str = f"[green]{mem_str}[/green]"

            temp_str = f"{gpu['temp']}°C"
            if gpu['temp'] > 80:
                temp_str = f"[red]{temp_str}[/red]"
            elif gpu['temp'] > 70:
                temp_str = f"[yellow]{temp_str}[/yellow]"
            else:
                temp_str = f"[green]{temp_str}[/green]"

            gpu_table.add_row(
                f"{gpu['id']}",
                gpu['name'][:20],
                util_str,
                mem_str,
                temp_str,
                status
            )

        console.print(gpu_table)
        console.print()

    # Running Processes - Compact
    if processes:
        proc_table = Table(show_header=True, header_style="bold yellow", box=None, padding=0)
        proc_table.add_column("PID", style="cyan", no_wrap=True)
        proc_table.add_column("CPU%", justify="right", no_wrap=True)
        proc_table.add_column("Mem%", justify="right", no_wrap=True)
        proc_table.add_column("Command", style="white")

        for proc in processes[:3]:  # Top 3 only
            cpu = float(proc['cpu'])
            cpu_str = f"{cpu:.0f}%"
            if cpu > 90:
                cpu_str = f"[red]{cpu_str}[/red]"
            elif cpu > 50:
                cpu_str = f"[yellow]{cpu_str}[/yellow]"
            else:
                cpu_str = f"[green]{cpu_str}[/green]"

            # Truncate command to fit
            cmd = proc['cmd']
            if len(cmd) > 60:
                cmd = cmd[:57] + "..."

            proc_table.add_row(
                proc['pid'],
                cpu_str,
                f"{float(proc['mem']):.1f}%",
                cmd
            )

        console.print(proc_table)
        console.print()
    else:
        console.print("[dim]⚙️  No training processes running[/dim]")
        console.print()

    # Experiments - Compact
    if experiments:
        exp_table = Table(show_header=True, header_style="bold cyan", box=None, padding=0)
        exp_table.add_column("Experiment", style="white")
        exp_table.add_column("Status", no_wrap=True)
        exp_table.add_column("Time", no_wrap=True, style="dim")
        exp_table.add_column("Metric", no_wrap=True)
        exp_table.add_column("Value", justify="right", no_wrap=True)
        exp_table.add_column("File", style="dim cyan", no_wrap=True)

        # Sort by value desc
        experiments = sorted(experiments, key=lambda x: x.get('value') or 0, reverse=True)

        for exp in experiments[:5]:  # Top 5
            # Status emoji
            status = exp.get('status', 'unknown')
            if status == 'completed':
                status_str = '[green]✅[/green]'
            elif status in ['training', 'in_progress']:
                status_str = '[yellow]🔄[/yellow]'
            elif status == 'failed':
                status_str = '[red]❌[/red]'
            else:
                status_str = '[blue]⏸️[/blue]'

            # Value coloring
            value = exp.get('value', 0)
            if value is None or value == 0:
                value_str = "[dim]N/A[/dim]"
            elif value > 0.5:
                value_str = f"[green bold]{value:.2f}[/green bold]"
            elif value > 0.1:
                value_str = f"[yellow]{value:.2f}[/yellow]"
            else:
                value_str = f"[red]{value:.2f}[/red]"

            # Shorten file path for display
            file_path = exp.get('file', '')
            file_short = file_path.replace('experiments/', '') if file_path else 'N/A'

            exp_table.add_row(
                exp['branch'][:30],
                status_str,
                exp.get('time_ago', '')[:10],
                exp.get('metric', 'N/A')[:15],
                value_str,
                file_short[:40]
            )

        console.print(exp_table)

        # Show recent notes from running experiments
        notes_shown = 0
        for exp in experiments[:3]:  # Only from top 3
            if exp.get('status') in ['running', 'in_progress']:
                # Load full experiment data to get notes
                exp_file = exp.get('file', '')
                if exp_file and os.path.exists(exp_file):
                    try:
                        import json
                        with open(exp_file, 'r') as f:
                            exp_data = json.load(f)
                            notes = exp_data.get('notes', [])
                            if notes:
                                # Show last 2 notes
                                console.print(f"\n[dim cyan]Notes from {exp['branch']}:[/dim cyan]")
                                for note in notes[-2:]:
                                    time_str = note.get('time', '')[:16]  # Just date+time
                                    text = note.get('text', '')
                                    console.print(f"  [dim]{time_str}[/dim] {text}")
                                notes_shown += 1
                    except:
                        pass

        if notes_shown > 0:
            console.print()
    else:
        console.print("[dim]🔬 No experiments tracked yet[/dim]")

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
