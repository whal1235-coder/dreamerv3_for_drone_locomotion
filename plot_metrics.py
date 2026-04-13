"""
Plot training metrics from metrics.jsonl (DreamerV3 or PPO).

Usage:
  python plot_metrics.py <logdir> [--out path/to/output.png] [--smooth N]
"""

import argparse
import json
import pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


COLOR = 'steelblue'


def smooth(values, w):
    arr = np.array(values, dtype=float)
    s = np.convolve(arr, np.ones(w), mode='full')[:len(arr)]
    counts = np.minimum(np.arange(1, len(arr) + 1), w)
    return s / counts, list(range(len(arr)))


def plot_series(ax, steps, values, color, title, w=None):
    ax.plot(steps, values, color=color, alpha=0.25)
    if w and len(values) > 1:
        s, idx = smooth(values, w)
        ax.plot([steps[i] for i in idx], s, color=color)
    ax.set_title(title)
    ax.set_xlabel('step')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--smooth', type=int, default=10)
    args = parser.parse_args()

    logdir = pathlib.Path(args.logdir)
    metrics_path = logdir / 'metrics.jsonl'
    assert metrics_path.exists(), f"Not found: {metrics_path}"

    data = [json.loads(l) for l in open(metrics_path)]
    w = args.smooth

    # ---- collect keys by group --------------------------------------------------
    episode_keys  = sorted({k for d in data for k in d if k.startswith('episode/')})
    epstats_keys  = sorted({k for d in data for k in d
                            if k.startswith('epstats/log/') and k.endswith('/avg')})
    loss_keys     = sorted({k for d in data for k in d if k.startswith('train/loss/')})
    fps_keys      = sorted({k for d in data for k in d if k.startswith('fps/')})

    groups = []
    if episode_keys:
        groups.append(('Episode', episode_keys + fps_keys))
    if epstats_keys:
        groups.append(('Reward', epstats_keys))
    if loss_keys:
        groups.append(('Loss', loss_keys))

    # ---- layout: each group is one row, max 4 per row ---------------------------
    n_cols = 4
    rows = []
    for group_name, keys in groups:
        for i in range(0, len(keys), n_cols):
            rows.append((group_name, keys[i:i + n_cols]))

    n_rows = max(len(rows), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    cur_step = max((d['step'] for d in data), default=0)
    fig.suptitle(f'{logdir.name}  (step {cur_step:,})', fontsize=13)

    score_entries = None
    used = set()
    for row_idx, (group_name, keys) in enumerate(rows):
        for col_idx, key in enumerate(keys):
            ax = axes[row_idx, col_idx]
            entries = [(d['step'], d[key]) for d in data if key in d]
            if not entries:
                ax.set_title(key.split('/')[-2] if '/' in key else key)
                ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            steps_  = [e[0] for e in entries]
            values_ = [e[1] for e in entries]
            label   = key.split('/')[-2] if key.endswith('/avg') else key.split('/')[-1]
            plot_series(ax, steps_, values_, COLOR, f'{group_name} / {label}', w=w)
            used.add((row_idx, col_idx))
            if key == 'episode/score':
                score_entries = (steps_, values_)

        # hide unused subplots in this row
        for col_idx in range(len(keys), n_cols):
            axes[row_idx, col_idx].axis('off')

    plt.tight_layout()
    out = args.out or str(logdir / 'loss.png')
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')
    plt.close(fig)

    # save score-only panel
    if score_entries is not None:
        fig_s, ax_s = plt.subplots(figsize=(6, 4.5))
        plot_series(ax_s, score_entries[0], score_entries[1], COLOR, 'Episode / score', w=w)
        fig_s.suptitle(f'{logdir.name}  (step {cur_step:,})', fontsize=13)
        plt.tight_layout()
        score_out = str(logdir / 'score.png')
        plt.savefig(score_out, dpi=150)
        print(f'Saved: {score_out}')
        plt.close(fig_s)


if __name__ == '__main__':
    main()
