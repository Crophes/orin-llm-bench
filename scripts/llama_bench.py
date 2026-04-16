#!/usr/bin/env python3
import os
import sys
import argparse
import datetime
import subprocess
import resource
import csv

OUTPUT = "/data/ram_usage.csv"

parser = argparse.ArgumentParser()
parser.add_argument("--model","-m", default="/models/meta-llama-3.1-8b-instruct.f16.gguf")
args = parser.parse_args()


cmd = [
   "/data/llama.cpp/build/bin/llama-bench",
    "-m", args.model,
    "-ngl", "99",
    "-fa", "1",
    "-o", "csv"
]

print("Running:", " ".join(cmd))

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

stdout_lines = []
while True:
    line = proc.stdout.readline()
    if not line:
        break
    stdout_lines.append(line.rstrip())

proc.wait()

stderr = proc.stderr.read()
if stderr:
    print(stderr, file=sys.stderr)

prefill_rate = None
prefill_stddev = None

decode_rate = None
decode_stddev = None
n_prompt = n_gen = ""

reader = csv.DictReader(stdout_lines)

for row in reader:
    n_prompt_val = int(row["n_prompt"])
    n_gen_val = int(row["n_gen"])

    # Prefill
    if n_prompt_val > 0 and n_gen_val == 0:
        n_prompt = n_prompt_val
        prefill_rate = float(row["avg_ts"])
        prefill_stddev = float(row["stddev_ts"])

    # Decode
    elif n_prompt_val == 0 and n_gen_val > 0:
        n_gen = n_gen_val
        decode_rate = float(row["avg_ts"])
        decode_stddev = float(row["stddev_ts"])


# RAM usage of self is not included since model/llama-bench operates only as child process
memory_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0  # ru_maxrss is in KB

is_new_file = not os.path.isfile(OUTPUT)
with open(OUTPUT, "a") as f:
    if is_new_file:
        f.write("timestamp,model,pp,tg,prefill_rate,decode_rate,prefill_stddev,decode_stddev,memory in MB\n")
    f.write(
        f"{datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')},"
        f"{os.path.basename(args.model)},"
        f"{n_prompt},{n_gen},"
        f"{prefill_rate},{decode_rate},"
        f"{prefill_stddev},{decode_stddev},"
        f"{memory_mb}\n"
    )


print(f"Saved results to: {OUTPUT}\n")

# build_commit,build_number,cpu_info,gpu_info,backends,model_filename,model_type,model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_mask,cpu_strict,poll,type_k,type_v,n_gpu_layers,n_cpu_moe,split_mode,main_gpu,no_kv_offload,flash_attn,devices,tensor_split,tensor_buft_overrides,use_mmap,use_direct_io,embeddings,no_op_offload,no_host,n_prompt,n_gen,n_depth,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts
# "287a33017","7772","ARMv8 Processor rev 1 (v8l)","Orin","CUDA","/models/meta-llama-3.1-8b-instruct.f16.gguf","llama 8B F16","16061055232","8030261312","2048","512","4","0x0","0","50","f16","f16","99","0","layer","0","0","1","auto","0.00","none","0","0","0","0","0","512","0","0","2026-04-15T23:13:05Z","2878558047","294642","177.866833","0.018056"
# "287a33017","7772","ARMv8 Processor rev 1 (v8l)","Orin","CUDA","/models/meta-llama-3.1-8b-instruct.f16.gguf","llama 8B F16","16061055232","8030261312","2048","512","4","0x0","0","50","f16","f16","99","0","layer","0","0","1","auto","0.00","none","0","0","0","0","0","0","128","0","2026-04-15T23:13:22Z","37637646806","4352045","3.400850","0.000393"
