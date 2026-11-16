#!/usr/bin/env python3
"""Compare first-text-token latency of uncached and checkpoint-enabled servers."""
import argparse
import json
import pathlib
import statistics
import time
import urllib.request


def sample(url, args):
    body = {'messages': [{'role': 'user', 'content': args.prompt}], 'temperature': 0,
            'repeat_penalty': 1, 'max_tokens': args.tokens, 'stream': True}
    req = urllib.request.Request(url.rstrip('/') + '/v1/chat/completions',
                                 data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    result = {'text': '', 'first_token_seconds': None}
    done = False
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == 'data: [DONE]': done = True
            elif line.startswith('data: '):
                chunk = json.loads(line[6:])
                text = chunk['choices'][0].get('delta', {}).get('content', '')
                if text and result['first_token_seconds'] is None:
                    result['first_token_seconds'] = time.monotonic() - started
                result['text'] += text
                for key in ['timings', 'usage', 'model']:
                    if key in chunk: result[key] = chunk[key]
    result['wall_seconds'] = time.monotonic() - started
    if not done or result['first_token_seconds'] is None or 'timings' not in result:
        raise RuntimeError('incomplete stream or missing timing data')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_url', help='server started with --no-prefix-cache')
    parser.add_argument('candidate_url', help='server with default prefix checkpointing')
    parser.add_argument('--prompt', default='Reply with the word Paris.')
    parser.add_argument('--tokens', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    if min(args.tokens, args.repeats, args.timeout) <= 0: parser.error('counts and timeout must be positive')
    warmup = {role: sample(getattr(args, role + '_url'), args) for role in ['baseline', 'candidate']}
    rows = []
    for i in range(args.repeats):
        order = ['baseline', 'candidate'] if i % 2 == 0 else ['candidate', 'baseline']
        for role in order:
            row = sample(getattr(args, role + '_url'), args)
            row['role'] = role
            cached = row['timings']['cached_tokens']
            if (role == 'baseline' and cached != 0) or (role == 'candidate' and cached <= 0):
                raise RuntimeError('cache mode did not match the requested comparison; inspect server flags and concurrent requests')
            rows.append(row)
            print(f'{role}: first token {row["first_token_seconds"]:.2f}s, cached {cached}', flush=True)
    reference = warmup['baseline']['text']
    if any(row['text'] != reference for row in [*warmup.values(), *rows]):
        raise RuntimeError('completion text differs between cached and fresh replay')
    summary = {role: {'median_first_token_seconds': statistics.median(row['first_token_seconds'] for row in rows if row['role'] == role),
                      'median_generation_tokens_per_second': statistics.median(row['timings']['predicted_per_second'] for row in rows if row['role'] == role)}
               for role in ['baseline', 'candidate']}
    result = {'prompt': args.prompt, 'warmup': warmup, 'samples': rows, 'summary': summary, 'outputs_equal': True}
    if args.output: args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
