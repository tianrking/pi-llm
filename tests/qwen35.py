#!/usr/bin/env python3
"""Qwen3.5 tokenizer fixtures and state isolation across HTTP requests."""
import argparse
import json
import pathlib
import socket
import subprocess
import time
import urllib.request
import urllib.error

ROOT = pathlib.Path(__file__).resolve().parents[1]


def request(port, prompt=None, stream=False, expect_cached=False, messages=None, stats=None):
    payload = {'messages': messages or [{'role': 'user', 'content': prompt}], 'max_tokens': 4,
               'temperature': 0, 'repeat_penalty': 1, 'stream': stream}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        if not stream:
            data = json.load(response)
            assert (data['timings']['cached_tokens'] > 0) == expect_cached, data
            if stats is not None:
                stats.update(data['timings']); stats['wall_seconds'] = time.monotonic() - started
            return data['choices'][0]['message']['content']
        text, done, timed = '', False, False
        for raw in response:
            line = raw.decode().strip()
            if line == 'data: [DONE]':
                done = True
            elif line.startswith('data: '):
                chunk = json.loads(line[6:])
                piece = chunk['choices'][0].get('delta', {}).get('content', '')
                if piece and not timed:
                    timed = True
                    if stats is not None: stats['first_token_seconds'] = time.monotonic() - started
                text += piece
                if 'timings' in chunk:
                    assert (chunk['timings']['cached_tokens'] > 0) == expect_cached, chunk
                    if stats is not None: stats.update(chunk['timings'])
        assert done and timed
        if stats is not None: stats['wall_seconds'] = time.monotonic() - started
        return text


def start_server(model, disabled=False):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    cmd = [str(ROOT / 'pillm'), '-m', str(model), '--serve', str(port), '-c', '256']
    if disabled: cmd.append('--no-prefix-cache')
    server = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(150):
            if server.poll() is not None: raise RuntimeError('test server exited')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1): return server, port
            except OSError: time.sleep(.1)
        raise RuntimeError('test server did not become ready')
    except BaseException:
        server.terminate(); server.wait(timeout=10); raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?', default='models/qwen3.5-2b-q4_0.gguf')
    args = parser.parse_args()
    model = pathlib.Path(args.model).resolve()
    subprocess.run([str(ROOT / 'tests/numerics'), '--state-check', str(model)], check=True, timeout=240)
    fixtures = json.loads((ROOT / 'tests/qwen35-tokenizer.json').read_text())
    for index, case in enumerate(fixtures['cases']):
        result = subprocess.run([str(ROOT / 'tests/numerics'), '--tokenize', str(model), case['text']],
                                capture_output=True, text=True, check=True, timeout=30)
        assert [int(x) for x in result.stdout.split()] == case['tokens'], index
    print('ok    six Qwen3.5 tokenizer fixtures match llama.cpp (marks, CJK, emoji, long pieces, added tokens)', flush=True)
    server, port = start_server(model)
    reference = None
    try:
        image = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
            data=json.dumps({'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'unused'}}]}]}).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            urllib.request.urlopen(image, timeout=5)
            raise AssertionError('text-only server accepted an image')
        except urllib.error.HTTPError as error:
            assert error.code == 400
        print('ok    unsupported image input is refused', flush=True)
        a = request(port, 'Reply with the word Paris.')
        assert a.strip(), 'empty completion'
        request(port, '请用中文说你好。')
        assert request(port, 'Reply with the word Paris.') == a, 'recurrent state leaked across requests'
        print('ok    A/B/A requests rebuild state and return the same greedy answer', flush=True)
        assert request(port, 'Reply with the word Paris.', stream=True, expect_cached=True) == a, 'streaming changed the answer'
        assert request(port, [{'type': 'text', 'text': 'Reply with the word Paris.'}], expect_cached=True) == a, 'text parts changed the answer'
        print('ok    streaming, non-streaming and text-part completions agree', flush=True)
        dialog = [{'role': 'user', 'content': 'Reply with the word Paris.'},
                  {'role': 'assistant', 'content': a},
                  {'role': 'user', 'content': 'Translate Paris to Chinese.'}]
        follow = request(port, messages=dialog, expect_cached=True)
        assert request(port, 'Reply with the word Paris.') == a, 'shorter history reused a longer checkpoint'
        reference, ref_port = start_server(model, disabled=True)
        assert request(ref_port, messages=dialog) == follow, 'cached multi-turn output differs from fresh replay'
        assert request(ref_port, messages=dialog) == follow, 'cache-disable flag did not keep requests fresh'
        print('ok    multi-turn checkpoint reuse matches uncached replay; shorter histories miss safely', flush=True)
        changed = [{'role': 'system', 'content': 'Be concise.'}, {'role': 'user', 'content': 'Say hello.'}]
        assert request(port, messages=changed) == request(ref_port, messages=changed), 'system change reused stale state'
        print('ok    system-prompt changes match a fresh replay', flush=True)
    finally:
        for child in [server, reference]:
            if child is None: continue
            child.terminate()
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()


if __name__ == '__main__':
    main()
