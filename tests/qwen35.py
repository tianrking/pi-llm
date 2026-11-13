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


def request(port, prompt, stream=False):
    payload = {'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 4,
               'temperature': 0, 'repeat_penalty': 1, 'stream': stream}
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        raw = response.read().decode()
    if not stream:
        data = json.loads(raw)
        assert data['timings']['cached_tokens'] == 0, data
        return data['choices'][0]['message']['content']
    assert raw.rstrip().endswith('data: [DONE]'), raw
    text = ''
    for line in raw.splitlines():
        if line.startswith('data: ') and line != 'data: [DONE]':
            chunk = json.loads(line[6:])
            text += chunk['choices'][0].get('delta', {}).get('content', '')
            if 'timings' in chunk:
                assert chunk['timings']['cached_tokens'] == 0, chunk
    return text


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
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([str(ROOT / 'pillm'), '-m', str(model), '--serve', str(port), '-c', '256'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(150):
            if server.poll() is not None:
                raise RuntimeError('test server exited')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise RuntimeError('test server did not become ready')
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
        assert request(port, 'Reply with the word Paris.', stream=True) == a, 'streaming changed the answer'
        assert request(port, [{'type': 'text', 'text': 'Reply with the word Paris.'}]) == a, 'text parts changed the answer'
        print('ok    streaming, non-streaming and text-part completions agree', flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == '__main__':
    main()
