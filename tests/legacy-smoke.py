#!/usr/bin/env python3
"""Exercise the existing qwen2/llama loaders without downloading old models."""
import pathlib
import random
import re
import struct
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def string(value):
    value = value.encode()
    return struct.pack('<Q', len(value)) + value


def make_model(path, arch):
    meta = []

    def scalar(key, kind, data):
        meta.append(string(key) + struct.pack('<I', kind) + data)

    def integer(key, value):
        scalar(key, 4, struct.pack('<I', value))

    scalar('general.architecture', 8, string(arch))
    for key, value in [('block_count', 1), ('embedding_length', 32), ('feed_forward_length', 64),
                       ('attention.head_count', 4), ('attention.head_count_kv', 2)]:
        integer(f'{arch}.{key}', value)
    scalar(f'{arch}.attention.layer_norm_rms_epsilon', 6, struct.pack('<f', 1e-6))
    scalar('tokenizer.ggml.pre', 8, string('qwen2' if arch == 'qwen2' else 'llama-bpe'))
    vocab = []
    extra = 0
    for byte in range(256):
        keep = 33 <= byte <= 126 or 161 <= byte <= 172 or 174 <= byte <= 255
        vocab.append(chr(byte if keep else 256 + extra))
        extra += not keep
    scalar('tokenizer.ggml.tokens', 9, struct.pack('<IQ', 8, 256) + b''.join(string(x) for x in vocab))
    scalar('tokenizer.ggml.token_type', 9, struct.pack('<IQ', 5, 256) + struct.pack('<256i', *([1] * 256)))
    scalar('tokenizer.ggml.merges', 9, struct.pack('<IQ', 8, 0))
    tensors = []
    rng = random.Random(17)

    def tensor(name, dims, kind):
        count = 1
        for dim in dims:
            count *= dim
        data = struct.pack('<' + 'f' * count, *([1] * count)) if kind == 0 else b''.join(
            struct.pack('<e', .0125) + bytes(rng.randrange(256) for _ in range(16)) for _ in range(count // 32))
        tensors.append((name, dims, kind, data))

    tensor('token_embd.weight', [32, 256], 2)
    tensor('output_norm.weight', [32], 0)
    for name in ['attn_norm', 'ffn_norm']:
        tensor(f'blk.0.{name}.weight', [32], 0)
    for name, dims in [('attn_q', [32, 32]), ('attn_k', [32, 16]), ('attn_v', [32, 16]),
                       ('attn_output', [32, 32]), ('ffn_gate', [32, 64]), ('ffn_up', [32, 64]), ('ffn_down', [64, 32])]:
        tensor(f'blk.0.{name}.weight', dims, 2)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(meta)) + b''.join(meta)
    payload = b''
    for name, dims, kind, data in tensors:
        payload += b'\0' * (-len(payload) % 32)
        header += string(name) + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims)
        header += struct.pack('<IQ', kind, len(payload))
        payload += data
    header += b'\0' * (-len(header) % 32)
    path.write_bytes(header + payload)


def main():
    with tempfile.TemporaryDirectory(prefix='pillm-legacy-') as directory:
        for arch in ['qwen2', 'llama']:
            model = pathlib.Path(directory) / f'{arch}.gguf'
            make_model(model, arch)
            base = [str(ROOT / 'pillm'), '-m', str(model), '--raw', '-p', 'abc', '-c', '64',
                    '-n', '3', '--temp', '0', '--tokens']
            a = subprocess.run(base + ['-t', '1', '-tb', '1'], capture_output=True, check=True, timeout=20)
            b = subprocess.run(base + ['-t', '2', '-tb', '2'], capture_output=True, check=True, timeout=20)
            assert a.stdout == b.stdout, arch
            assert re.findall(rb'\[(\d+)\]', a.stderr) == re.findall(rb'\[(\d+)\]', b.stderr), arch
            empty = subprocess.run([str(ROOT / 'pillm'), '-m', str(model), '--raw', '-p', '', '-c', '64'], capture_output=True, timeout=20)
            assert empty.returncode == 1, arch
            print(f'ok    synthetic {arch} GGUF loads, decodes consistently and refuses an empty raw prompt', flush=True)


if __name__ == '__main__':
    main()
