"""
Multi-split Boxoban benchmark for HalfWeg.

Usage:
    python -u benchmark_boxoban.py configs/benchmark_boxoban_splits.yaml \\
        --levels-root /path/to/boxoban-levels \\
        --output-csv results/benchmark_results.csv

For each split defined in the config the script:
 1. Enumerates level files (optionally sampling up to max_files).
 2. Runs closed_loop_replan evaluation and collects per-game stats.
 3. Aggregates across files and prints a summary table + saves CSV.
"""

import csv
import glob
import os
import pprint
import random
import sys

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Make sure the src/ directory is on the path when running directly
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import environments
import helpers
from hw_impl import evaluate, hw_experience_replay, model_mgmt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_levels_root(config: dict, cli_levels_root: str | None) -> str:
    """Return the levels root from CLI override > config field > cwd."""
    if cli_levels_root:
        return os.path.abspath(cli_levels_root)
    cfg_root = config.get('benchmark', {}).get('levels_root', None)
    if cfg_root:
        return os.path.abspath(cfg_root)
    # Guess: walk up from this file looking for boxoban-levels/
    candidate = os.path.join(_SRC_DIR, '..', '..', 'boxoban-levels')
    if os.path.isdir(candidate):
        return os.path.abspath(candidate)
    return os.getcwd()


def _collect_split_files(split: dict, levels_root: str) -> list[str]:
    """Return sorted list of .txt level files for a split definition."""
    pattern = split.get('levels_glob', split.get('levels', ''))
    if not os.path.isabs(pattern):
        pattern = os.path.join(levels_root, pattern)

    files = sorted(glob.glob(pattern))

    max_files = split.get('max_files', len(files))
    seed = split.get('file_sample_seed', 42)
    if max_files < len(files):
        rng = random.Random(seed)
        files = sorted(rng.sample(files, max_files))

    return files


def _aggregate_results(per_file_results: list[list[dict]]) -> dict:
    """Aggregate a flat list of per-policy result dicts (one per file) into totals."""
    if not per_file_results:
        return {}

    # Collect only 'towards' results (towards_or_away == True) for the primary policy
    rows = [r for file_res in per_file_results for r in file_res if r.get('towards_or_away') is True]
    if not rows:
        rows = [r for file_res in per_file_results for r in file_res]

    def _avg(key, default=float('nan')):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return float(np.mean(vals)) if vals else default

    def _sum(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return int(np.sum(vals)) if vals else 0

    total_games = _sum('games_cnt')
    total_solved = _sum('solved_count')

    return {
        'files_evaluated': len(per_file_results),
        'total_games': total_games,
        'total_solved': total_solved,
        'solve_rate': total_solved / total_games if total_games > 0 else float('nan'),
        'timeout_rate': _avg('timeout_rate'),
        'mse_mean': _avg('mse_mean'),
        'proposed_plan_length_mean': _avg('proposed_plan_length_mean'),
        'solved_plan_length_mean': _avg('solved_plan_length_mean'),
        'solved_plan_length_p50': _avg('solved_plan_length_p50'),
        'solved_plan_length_p90': _avg('solved_plan_length_p90'),
        'avg_replan_count': _avg('avg_replan_count'),
        'avg_box_push_count': _avg('avg_box_push_count'),
        'partial_progress_unsolved_mean': _avg('partial_progress_unsolved_mean'),
    }


def _print_summary_table(split_summaries: list[dict]):
    cols = [
        ('split_name', 20),
        ('files_evaluated', 6),
        ('total_games', 6),
        ('solve_rate', 9),
        ('timeout_rate', 10),
        ('solved_plan_length_p50', 8),
        ('solved_plan_length_p90', 8),
        ('avg_replan_count', 10),
        ('avg_box_push_count', 10),
        ('partial_progress_unsolved_mean', 12),
        ('mse_mean', 9),
    ]

    header = '  '.join(name.ljust(w) for name, w in cols)
    sep = '  '.join('-' * w for _, w in cols)
    print('\n' + '=' * len(sep))
    print('BENCHMARK SUMMARY')
    print('=' * len(sep))
    print(header)
    print(sep)

    for row in split_summaries:
        def _fmt(key, width):
            v = row.get(key, 'n/a')
            if isinstance(v, float):
                s = f'{v:.4f}' if not np.isnan(v) else 'n/a'
            else:
                s = str(v)
            return s.ljust(width)

        print('  '.join(_fmt(name, w) for name, w in cols))

    print('=' * len(sep) + '\n')


def _save_csv(split_summaries: list[dict], output_path: str):
    if not split_summaries:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    keys = list(split_summaries[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(split_summaries)
    print(f'>>> Benchmark CSV saved to: {output_path}')


# ---------------------------------------------------------------------------
# Main benchmark logic
# ---------------------------------------------------------------------------

def run_benchmark(config: dict, device, cli_levels_root: str | None, output_csv: str | None):
    levels_root = _resolve_levels_root(config, cli_levels_root)
    print(f'>>> Levels root: {levels_root}')

    # Load model once
    model_keeper = model_mgmt.ModelKeeper(config)
    model_keeper.to(device)
    envs_sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)
    print('>>> Checkpoint loaded')

    # Resolve policies
    policy_names = evaluate.get_policies(model_keeper, config['evaluate']['policies'])
    policies = [
        evaluate.get_policy(name, model_keeper, envs_sampler)
        for name in policy_names
    ]

    method = config['evaluate']['method']
    towards_or_away_array = (
        [True, False] if config['evaluate']['towards_or_away'] == 'both'
        else [False] if config['evaluate']['towards_or_away'] == 'away'
        else [True]
    )

    splits = config['benchmark']['splits']
    split_summaries = []

    for split in splits:
        split_name = split['name']
        n_games = split.get('n_games', config['evaluate'].get('n_games_to_solve', 10))

        files = _collect_split_files(split, levels_root)
        if not files:
            print(f'>>> WARNING: No files found for split "{split_name}" — skipping')
            continue

        print(f'\n>>> Split "{split_name}": {len(files)} file(s), {n_games} games/file')

        per_file_results = []

        for file_i, filepath in enumerate(files):
            print(f'  File {file_i + 1}/{len(files)}: {os.path.basename(filepath)}')

            env_config = dict(config['env'])
            env_config['levels'] = filepath

            try:
                envs_manager = environments.create_envs_manager(env_config)
            except Exception as e:
                print(f'    ERROR loading levels: {e}')
                continue

            file_results = evaluate.validate_puzzle_solving__impl(
                config=config,
                method=method,
                device=device,
                envs_manager=envs_manager,
                model_keeper=model_keeper,
                n_games_to_solve=n_games,
                policies=policies,
                towards_or_away_array=towards_or_away_array,
                tensorboard=None,
            )
            per_file_results.append(file_results)

        agg = _aggregate_results(per_file_results)
        agg['split_name'] = split_name
        split_summaries.append(agg)
        print(f'  Split "{split_name}" done — solve_rate={agg.get("solve_rate", "n/a"):.4f}  '
              f'games={agg.get("total_games", 0)}')

    _print_summary_table(split_summaries)

    csv_path = output_csv or config.get('benchmark', {}).get('output_csv', None)
    if csv_path:
        _save_csv(split_summaries, csv_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Multi-split Boxoban benchmark for HalfWeg')
    parser.add_argument('config', help='Path to benchmark YAML config')
    parser.add_argument('--levels-root', default=None,
                        help='Root directory for boxoban-levels (overrides config/auto-detect)')
    parser.add_argument('--output-csv', default=None,
                        help='Path to write CSV summary (overrides config)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.set_num_threads(1)
    torch.autograd.set_detect_anomaly(False)

    if config['infra'].get('device') in (None, 'cpu'):
        device = 'cpu'
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    pprint.pprint(config)

    run_benchmark(config, device, cli_levels_root=args.levels_root, output_csv=args.output_csv)


if __name__ == '__main__':
    main()
