"""
Generate HTML report from benchmark results.
"""

import json
import argparse
from pathlib import Path
import base64
from datetime import datetime
from string import Template

HTML_TEMPLATE = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TensorMesh Benchmark Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 30px;
        }
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin-top: 0;
            color: #666;
            font-size: 14px;
            text-transform: uppercase;
        }
        .card .value {
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
        }
        .card .unit {
            font-size: 14px;
            color: #999;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #4CAF50;
            color: white;
        }
        tr:hover {
            background: #f5f5f5;
        }
        .fastest {
            background: #c8e6c9 !important;
            font-weight: bold;
        }
        .slowest {
            background: #ffcdd2 !important;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .image-grid img {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #999;
            font-size: 12px;
            text-align: center;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge-cuda { background: #2196F3; color: white; }
        .badge-cpu { background: #FF9800; color: white; }
    </style>
</head>
<body>
    <h1>TensorMesh Benchmark Report</h1>
    <p><strong>Generated:</strong> %(timestamp)s</p>
    
    <h2>Summary</h2>
    <div class="summary-cards">
        %(summary_cards)s
    </div>
    
    <h2>Performance Table</h2>
    %(table)s
    
    <h2>Visualizations</h2>
    <div class="image-grid">
        %(images)s
    </div>
    
    <div class="footer">
        TensorMesh Benchmark Suite | Generated at %(timestamp)s
    </div>
</body>
</html>
""")


def load_jsonl(filepath):
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_loss_report(data, image_paths):
    """Generate report for loss comparison experiment."""
    
    from collections import defaultdict
    import statistics
    
    # Aggregate data
    stats = defaultdict(lambda: {
        'fwd_times': [], 'bwd_times': [], 'total_times': [],
        'memories': [], 'dofs': set()
    })
    
    for record in data:
        name = record['loss_name']
        stats[name]['fwd_times'].append(record['time_forward'])
        stats[name]['bwd_times'].append(record['time_backward'])
        stats[name]['total_times'].append(record['time_total'])
        stats[name]['memories'].append(record.get('memory_peak', 0))
        stats[name]['dofs'].add(record['dof'])
    
    # Calculate summary
    summary = {}
    for name, s in stats.items():
        summary[name] = {
            'mean_fwd': statistics.mean(s['fwd_times']) * 1000,  # ms
            'mean_bwd': statistics.mean(s['bwd_times']) * 1000,
            'mean_total': statistics.mean(s['total_times']) * 1000,
            'mean_mem': statistics.mean(s['memories']),
            'dofs': sorted(s['dofs'])
        }
    
    # Find fastest
    fastest = min(summary.items(), key=lambda x: x[1]['mean_total'])
    slowest = max(summary.items(), key=lambda x: x[1]['mean_total'])
    speedup = slowest[1]['mean_total'] / fastest[1]['mean_total']
    
    # Generate cards
    cards = f"""
        <div class="card">
            <h3>Fastest Loss</h3>
            <div class="value">{fastest[0]}</div>
            <div class="unit">{fastest[1]['mean_total']:.2f} ms avg</div>
        </div>
        <div class="card">
            <h3>Speedup Factor</h3>
            <div class="value">{speedup:.2f}×</div>
            <div class="unit">vs {slowest[0]}</div>
        </div>
        <div class="card">
            <h3>Tested Configs</h3>
            <div class="value">{len(summary)}</div>
            <div class="unit">loss functions</div>
        </div>
        <div class="card">
            <h3>DOF Range</h3>
            <div class="value">{min([min(s['dofs']) for s in summary.values()])}</div>
            <div class="unit">to {max([max(s['dofs']) for s in summary.values()])}</div>
        </div>
    """
    
    # Generate table
    rows = []
    for name, s in sorted(summary.items(), key=lambda x: x[1]['mean_total']):
        row_class = "fastest" if name == fastest[0] else "slowest" if name == slowest[0] else ""
        rows.append(f"""
            <tr class="{row_class}">
                <td><strong>{name}</strong></td>
                <td>{s['mean_fwd']:.2f}</td>
                <td>{s['mean_bwd']:.2f}</td>
                <td>{s['mean_total']:.2f}</td>
                <td>{s['mean_mem']:.1f}</td>
                <td>{speedup:.2f}×</td>
            </tr>
        """)
    
    table = f"""
        <table>
            <tr>
                <th>Loss Function</th>
                <th>Forward (ms)</th>
                <th>Backward (ms)</th>
                <th>Total (ms)</th>
                <th>Memory (MB)</th>
                <th>Speedup</th>
            </tr>
            {''.join(rows)}
        </table>
    """
    
    # Embed images
    images = []
    for path in image_paths:
        if path.exists():
            with open(path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            images.append(f'<img src="data:image/png;base64,{img_data}" alt="{path.stem}">')
    
    return cards, table, ''.join(images)


def generate_fem_report(data, image_paths):
    """Generate report for FEM comparison experiment."""
    
    from collections import defaultdict
    import statistics
    
    # Aggregate data
    stats = defaultdict(lambda: {
        'assemble_times': [], 'solve_times': [], 'total_times': [],
        'errors': [], 'memories': []
    })
    
    for record in data:
        name = record['library']
        stats[name]['assemble_times'].append(record['time_assemble'])
        stats[name]['solve_times'].append(record['time_solve'])
        stats[name]['total_times'].append(record['time_total'])
        stats[name]['memories'].append(record.get('memory_peak', 0))
        if record.get('l2_error'):
            stats[name]['errors'].append(record['l2_error'])
    
    # Calculate summary
    summary = {}
    for name, s in stats.items():
        summary[name] = {
            'mean_assemble': statistics.mean(s['assemble_times']) * 1000,
            'mean_solve': statistics.mean(s['solve_times']) * 1000,
            'mean_total': statistics.mean(s['total_times']) * 1000,
            'mean_mem': statistics.mean(s['memories']),
            'mean_error': statistics.mean(s['errors']) if s['errors'] else 0
        }
    
    fastest = min(summary.items(), key=lambda x: x[1]['mean_total'])
    
    # Generate cards
    cards = f"""
        <div class="card">
            <h3>Fastest Solver</h3>
            <div class="value">{fastest[0]}</div>
            <div class="unit">{fastest[1]['mean_total']:.2f} ms avg</div>
        </div>
        <div class="card">
            <h3>Solvers Tested</h3>
            <div class="value">{len(summary)}</div>
            <div class="unit">FEM libraries</div>
        </div>
        <div class="card">
            <h3>Best Accuracy</h3>
            <div class="value">{min(summary.items(), key=lambda x: x[1]['mean_error'])[0]}</div>
            <div class="unit">lowest L2 error</div>
        </div>
    """
    
    # Generate table
    rows = []
    for name, s in sorted(summary.items(), key=lambda x: x[1]['mean_total']):
        row_class = "fastest" if name == fastest[0] else ""
        rows.append(f"""
            <tr class="{row_class}">
                <td><strong>{name}</strong></td>
                <td>{s['mean_assemble']:.2f}</td>
                <td>{s['mean_solve']:.2f}</td>
                <td>{s['mean_total']:.2f}</td>
                <td>{s['mean_mem']:.1f}</td>
                <td>{s['mean_error']:.2e}</td>
            </tr>
        """)
    
    table = f"""
        <table>
            <tr>
                <th>Library</th>
                <th>Assemble (ms)</th>
                <th>Solve (ms)</th>
                <th>Total (ms)</th>
                <th>Memory (MB)</th>
                <th>L2 Error</th>
            </tr>
            {''.join(rows)}
        </table>
    """
    
    # Embed images
    images = []
    for path in image_paths:
        if path.exists():
            with open(path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            images.append(f'<img src="data:image/png;base64,{img_data}" alt="{path.stem}">')
    
    return cards, table, ''.join(images)


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--plot-dir", default="./plots",
                       help="Directory containing generated plots")
    parser.add_argument("--output", default="./benchmark_report.html",
                       help="Output HTML file")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}...")
    data = load_jsonl(args.input)
    
    plot_dir = Path(args.plot_dir)
    image_paths = list(plot_dir.glob("*.png")) if plot_dir.exists() else []
    
    # Detect experiment type
    if data and 'loss_name' in data[0]:
        print("Generating Loss Comparison Report...")
        cards, table, images = generate_loss_report(data, image_paths)
    else:
        print("Generating FEM Comparison Report...")
        cards, table, images = generate_fem_report(data, image_paths)
    
    # Generate HTML using string.Template
    html = HTML_TEMPLATE.substitute(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        summary_cards=cards,
        table=table,
        images=images
    )
    
    with open(args.output, 'w') as f:
        f.write(html)
    
    print(f"[OK] Report saved to: {args.output}")
    print(f"   Open in browser: file:///{Path(args.output).absolute()}")


if __name__ == "__main__":
    main()
