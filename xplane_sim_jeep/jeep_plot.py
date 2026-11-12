"""
Jeep Telemetry Plotter
======================
Auto-finds the latest CSV from jeep_logs/ and plots trajectory + speed.
Can also specify a particular CSV file.

Usage:
  python jeep_plot.py                              # auto-find latest CSV
  python jeep_plot.py jeep_logs/jeep_xxx.csv       # specify file
"""

import sys
import os
import glob
import math
import csv
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import numpy as np


LOG_DIR = "jeep_logs"


def find_latest_csv(log_dir=LOG_DIR):
    """Find the most recent jeep_*.csv in log_dir"""
    pattern = os.path.join(log_dir, "jeep_*.csv")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort()
    return files[-1]


def load_csv(filepath):
    """Load CSV into dict of numpy arrays"""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for key in reader.fieldnames:
            data[key] = []
        for row in reader:
            for key in reader.fieldnames:
                data[key].append(float(row[key]))
    for key in data:
        data[key] = np.array(data[key])
    return data


def plot_jeep_data(data, csv_path, save_png=True):
    """Plot trajectory, speed, heading, and control outputs"""
    
    t = data['time']
    x = data['local_x']
    z = data['local_z']
    speed = data['groundspeed']
    heading = data['heading']
    h_err = data['heading_err']
    rudder = data['rudder']
    throttle = data['throttle']
    
    # Relative coordinates (origin = start point)
    x_rel = x - x[0]
    z_rel = z - z[0]
    
    # X-Plane: local_z is negative toward south -> negate so south = up on plot
    north = -z_rel
    east = x_rel
    
    # Statistics
    dx = np.diff(east)
    dn = np.diff(north)
    total_distance = np.sum(np.sqrt(dx**2 + dn**2))
    avg_speed = np.mean(speed)
    max_speed = np.max(speed)
    
    # ═══════════════════════════════════════════════
    #  Plot: 2x2 layout
    # ═══════════════════════════════════════════════
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f'Jeep Telemetry  |  {os.path.basename(csv_path)}',
                 fontsize=13, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)
    
    # ────────────────────────────────────────────
    # Top-left: Trajectory (color = speed)
    # ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    
    points = np.column_stack([east, north])
    segments = np.array([points[i:i+2] for i in range(len(points)-1)])
    
    norm = plt.Normalize(speed.min(), max(speed.max(), speed.min() + 0.1))
    lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=2.5)
    lc.set_array(speed[:-1])
    ax1.add_collection(lc)
    
    ax1.plot(east[0], north[0], 'o', color='#2ecc71', markersize=12,
             markeredgecolor='white', markeredgewidth=2, zorder=5, label='Start')
    ax1.plot(east[-1], north[-1], 's', color='#e74c3c', markersize=12,
             markeredgecolor='white', markeredgewidth=2, zorder=5, label='End')
    
    # Heading arrows
    arrow_interval = max(1, len(east) // 10)
    arrow_len = max(total_distance * 0.03, 1.0)
    for i in range(0, len(east), arrow_interval):
        hdg_rad = math.radians(heading[i])
        dx_a = math.sin(hdg_rad) * arrow_len
        dn_a = math.cos(hdg_rad) * arrow_len
        ax1.annotate('', xy=(east[i]+dx_a, north[i]+dn_a),
                     xytext=(east[i], north[i]),
                     arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    cbar = plt.colorbar(lc, ax=ax1, pad=0.02, shrink=0.8)
    cbar.set_label('Speed (m/s)', fontsize=9)
    
    ax1.set_xlabel('East (m)', fontsize=10)
    ax1.set_ylabel('North (m)', fontsize=10)
    ax1.set_title('Trajectory', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.autoscale()
    
    x_margin = max(5, (east.max() - east.min()) * 0.15)
    y_margin = max(5, (north.max() - north.min()) * 0.15)
    ax1.set_xlim(east.min() - x_margin, east.max() + x_margin)
    ax1.set_ylim(north.min() - y_margin, north.max() + y_margin)
    
    # ────────────────────────────────────────────
    # Top-right: Speed vs Time
    # ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.fill_between(t, 0, speed, alpha=0.15, color='#3498db')
    ax2.plot(t, speed, color='#2980b9', linewidth=1.5, label='Ground Speed')
    ax2.axhline(y=avg_speed, color='#e67e22', linestyle='--', linewidth=1.5,
                label=f'Avg: {avg_speed:.2f} m/s')
    
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Speed (m/s)', fontsize=10)
    ax2.set_title('Ground Speed', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, max(max_speed * 1.3, 1.0))
    
    stats_text = (
        f'Total Dist: {total_distance:.1f} m\n'
        f'Duration: {t[-1]:.1f} s\n'
        f'Avg: {avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h)\n'
        f'Max: {max_speed:.2f} m/s ({max_speed*3.6:.1f} km/h)'
    )
    ax2.text(0.02, 0.97, stats_text, transform=ax2.transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))
    
    # ────────────────────────────────────────────
    # Bottom-left: Heading & Heading Error
    # ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.plot(t, heading, color='#8e44ad', linewidth=1.5, label='Heading (deg)')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Heading (deg)', fontsize=10, color='#8e44ad')
    ax3.tick_params(axis='y', labelcolor='#8e44ad')
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(t, h_err, color='#e74c3c', linewidth=1.0, alpha=0.7, label='Heading Error (deg)')
    ax3_twin.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    ax3_twin.set_ylabel('Heading Error (deg)', fontsize=10, color='#e74c3c')
    ax3_twin.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax3.set_title('Heading Tracking', fontsize=12, fontweight='bold')
    ax3.set_xlim(t[0], t[-1])
    ax3.grid(True, alpha=0.3)
    
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # ────────────────────────────────────────────
    # Bottom-right: Control Outputs
    # ────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.plot(t, rudder, color='#2980b9', linewidth=1.2, label='Rudder (Steering)')
    ax4.plot(t, throttle, color='#27ae60', linewidth=1.2, label='Throttle')
    ax4.axhline(y=0, color='gray', linestyle=':', linewidth=0.8)
    
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Control Value', fontsize=10)
    ax4.set_title('Control Output', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(t[0], t[-1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save PNG
    if save_png:
        png_path = csv_path.replace('.csv', '.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Saved: {png_path}")
    
    plt.show()
    print("[Plot] Done")


def main():
    parser = argparse.ArgumentParser(
        description="Jeep telemetry plotter (auto-finds latest CSV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jeep_plot.py                              # auto-find latest CSV
  python jeep_plot.py jeep_logs/jeep_xxx.csv       # specify file
  python jeep_plot.py --no-save                    # don't save PNG
        """
    )
    parser.add_argument('csv_file', nargs='?', default=None,
                        help='CSV file path (auto-finds latest if not specified)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save PNG file')
    
    args = parser.parse_args()
    
    if args.csv_file:
        csv_path = args.csv_file
    else:
        csv_path = find_latest_csv()
    
    if csv_path is None or not os.path.exists(csv_path):
        print(f"[Error] No CSV file found")
        print(f"        Run jeep_controller.py first to generate data")
        print(f"        Or specify a file: python jeep_plot.py <csv_path>")
        sys.exit(1)
    
    print(f"\n[Plot] Loading: {csv_path}")
    data = load_csv(csv_path)
    print(f"[Plot] {len(data['time'])} samples, duration {data['time'][-1]:.1f} s")
    
    plot_jeep_data(data, csv_path, save_png=not args.no_save)


if __name__ == '__main__':
    main()