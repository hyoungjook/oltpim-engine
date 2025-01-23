#!/usr/bin/python3

import argparse
import pathlib
import subprocess
import tempfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpu-binary', type=str, required=True)
    parser.add_argument('--dump-file', type=str, required=True)
    return parser.parse_args()

def generate_trace(dpu_binary, dump_file, trace_dir):
    simulate_bin = str((pathlib.Path(__file__).parent / 'simulate_sample_dpu').absolute())
    dpu_binary_abs = str(pathlib.Path(dpu_binary).absolute())
    dump_file_abs = str(pathlib.Path(dump_file).absolute())
    trace_dir_abs = str(pathlib.Path(trace_dir).absolute())
    subprocess.run([
        simulate_bin,
        dpu_binary_abs,
        dump_file_abs,
        trace_dir_abs
    ])

def parse_trace(trace_dir):
    dputrace_bin = 'dputrace'
    trace_file = str((pathlib.Path(trace_dir) / 'trace-0000-00').absolute())
    result_bytes = subprocess.run([
        dputrace_bin,
        '-i', trace_file,
        '-no-color', '-no-tree'
    ], capture_output=True)
    result_str = result_bytes.stdout.decode('utf-8')
    instructions = result_str.split('\n')

    total_count = 0
    wram_count = 0
    mram_count = 0
    mram_avg_size = 0

    OP_LDMA_SDMA = {'ldma', 'sdma'}
    OP_LD_ST = {'lbs', 'lbu', 'ld', 'lhs', 'lhu', 'lw', 'sb', 'sd', 'sh', 'sw'}

    for instruction in instructions:
        tokens = instruction.split(' ')
        tokens = list(filter(bool, tokens))
        if len(tokens) < 2:
            continue
        # ['[00@0x80000010]', 'move', 'r23', '(0x00000100),', '256', '(0x100)']
        op = tokens[1]
        total_count += 1
        if op in OP_LDMA_SDMA:
            mram_count += 1
            imm = tokens[6]
            dma_size = 8 * (1 + int(imm))
            mram_avg_size += dma_size
        elif op in OP_LD_ST:
            wram_count += 1
    mram_avg_size = float(mram_avg_size) / mram_count

    return (total_count, wram_count, mram_count, mram_avg_size)

if __name__ == "__main__":
    args = parse_args()
    trace_dir = tempfile.TemporaryDirectory()
    generate_trace(args.dpu_binary, args.dump_file, trace_dir.name)
    total, wram, mram, mram_size = parse_trace(trace_dir.name)
    wram_ratio = float(wram) / total
    mram_ratio = float(mram) / total
    print(f'{wram_ratio} {mram_ratio} {mram_size}')
