#!/usr/bin/env python3
"""Interleave two pillm builds; compare generation speed, CPU cost and token IDs."""
import argparse
import hashlib
import json
import pathlib
import re
import resource
import statistics
import subprocess
import time


def run(binary, args):
    command = [str(binary), '-m', str(args.model), '--raw', '-p', args.prompt,
               '-n', str(args.tokens), '-c', str(args.context), '-t', str(args.threads),
               '-tb', str(args.batch_threads), '--temp', '0', '--repeat', '1', '--tokens', '--prof']
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, errors='replace', check=True, timeout=args.timeout)
    wall = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    match = re.search(r'gen (\d+) tok, ([\d.]+) tok/s', proc.stderr)
    if not match or int(match[1]) == 0:
        raise RuntimeError(f'no generated tokens: {proc.stderr}')
    return {'binary': str(binary), 'command': command, 'generation_tokens': int(match[1]),
            'generation_tokens_per_second': float(match[2]), 'wall_seconds': wall,
            'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            'token_ids': [int(x) for x in re.findall(r'\[(\d+)\]', proc.stderr)],
            'stdout': proc.stdout, 'stderr': proc.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=pathlib.Path)
    parser.add_argument('candidate', type=pathlib.Path)
    parser.add_argument('--model', type=pathlib.Path, default=pathlib.Path('models/qwen3.5-2b-q4_0.gguf'))
    parser.add_argument('--prompt', default='def fibonacci(n):')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--context', type=int, default=256)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--batch-threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=240)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    if min(args.tokens, args.context, args.threads, args.batch_threads, args.repeats, args.timeout) < 1:
        parser.error('counts and timeout must be positive')
    args.baseline = args.baseline.resolve()
    args.candidate = args.candidate.resolve()
    args.model = args.model.resolve()
    rows = []
    for pair in range(args.repeats):
        order = ['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']
        for role in order:
            sample = run(getattr(args, role), args)
            sample['role'] = role
            rows.append(sample)
            print(f'{role}: {sample["generation_tokens_per_second"]:.2f} token/s, {sample["cpu_seconds"]:.2f} CPU s/request', flush=True)
    result = {'binary_sha256': {role: hashlib.sha256(getattr(args, role).read_bytes()).hexdigest()
                               for role in ['baseline', 'candidate']}, 'samples': rows,
              'token_ids_equal': all(row['token_ids'] == rows[0]['token_ids'] for row in rows)}
    for role in ['baseline', 'candidate']:
        selected = [row for row in rows if row['role'] == role]
        result[role] = {'median_generation_tokens_per_second': statistics.median(row['generation_tokens_per_second'] for row in selected),
                        'median_cpu_seconds_per_request': statistics.median(row['cpu_seconds'] for row in selected)}
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ['baseline', 'candidate', 'token_ids_equal']}, indent=2))
    if not result['token_ids_equal']:
        raise SystemExit('generated token IDs differ; inspect the captured samples')


if __name__ == '__main__':
    main()
