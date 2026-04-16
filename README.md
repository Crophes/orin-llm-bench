# jetson-llm-bench

Benchmarking LLM inference performance on the NVIDIA Jetson AGX Orin —
evaluating quantization formats and framework differences between llama.cpp
and MLC across a diverse set of model families.

---

## Hardware & Software Requirements

- NVIDIA Jetson AGX Orin (64 GB recommended)
- JetPack 6.0, CUDA 12.8, Ubuntu 24.04
- [jetson-containers](https://github.com/dusty-nv/jetson-containers)
- HuggingFace account and access token for downloading gated models

Install jetson-containers:

```bash
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
pip install -r requirements.txt
```

The repository contains two host runner scripts (`llama_bench_runner.sh`
and `mlc_bench_runner.sh`/benchmark.sh with mlc_bench.py/benchmark.py) with `llama_bench.py located in their respective
container.

---

## System Configuration

All benchmarks were run with the system configured for maximum performance.

> **Warning:** Changing the power mode will restart the device.

Set maximum power mode and maximize clocks:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

To restore default operation afterwards:

```bash
sudo nvpmodel -m 1
```

---

## llama.cpp Benchmarks

### 1. Download GGUF models

Download GGUF models from HuggingFace using the `hf` CLI tool:

```bash
# Download a specific file
hf download <repository> <filename> --local-dir <download_directory>

# Examples
hf download SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF \
    meta-llama-3.1-8b-instruct.f16.gguf \
    --local-dir /path/to/models

hf download SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF \
    meta-llama-3.1-8b-instruct.Q4_0.gguf \
    --local-dir /path/to/models
```

Place all downloaded GGUF files in a single directory.
This directory will be referred to as `/path/to/models` throughout this guide.

### 2. Run the benchmarks

The runner script iterates over all `.gguf` files in your models directory
and benchmarks each one. Edit `llama_bench_runner.sh` and set `MODELS_DIR`
to your models directory, then run from the host:

```bash
bash llama_bench_runner.sh
```

To benchmark a single model directly:

```bash
jetson-containers run \
    -v /path/to/models:/models \
    $(autotag llama_cpp) \
    python3 llama_bench_memory.py --model /models/your-model.gguf
```

Results are saved to a CSV file. See the [Output](#output) section for details.

### 3. Running llama-bench manually

To run llama-bench directly inside the container for quick testing:

```bash
jetson-containers run \
    -v /path/to/models:/models \
    $(autotag llama_cpp) \
    llama-bench -m /models/your-model.gguf -ngl 99 -fa 1
```

`-ngl 99` offloads all layers to the GPU.\
`-fa 1` enables flash attention.
Use `-fa 0` to disable it.\
`-o csv/json` for detailed output

### 4. Using a specific version of llama.cpp

Some models require a newer version of llama.cpp than what the container
provides by default. For example, GPT-Oss 20B in MXFP4 will fail with the
following error on older versions:

```
gguf_init_from_file_impl: tensor 'blk.0.ffn_down_exps.weight' has invalid ggml type 39 (NONE)
```

To build a specific version, start the llama.cpp container and build inside it:

```bash
jetson-containers run \
    -v /path/to/models:/models \
    $(autotag llama_cpp)
```

Inside the container:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout <COMMIT_HASH>
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

To run benchmarks with the newly built version, navigate to the new
llama.cpp directory and use the local binary directly:

```bash
cd llama.cpp
./build/bin/llama-bench -m /models/your-model.gguf -ngl 99 -fa 1
```

The benchmarks in this repository used commit `c1b1876`.

### 5. Running the llama.cpp web UI

To chat with a model via a browser interface:

```bash
jetson-containers run \
    -p 8033:8033 \
    $(autotag llama_cpp) \
    llama-server \
        --hf Qwen/Qwen3-1.7B-GGUF \
        --jinja \
        --c 0 \
        --host 127.0.0.1 \
        --port 8033
```
Use `--hf <repository>` to download and run a model directly from HuggingFace,
or `-m /models/your-model.gguf` to use a local model file.\
`-c 0` sets the size of the prompt context to model default, instead of 4096.\
`--jinja` uses the jinja template for chat.\
Open `http://127.0.0.1:8033` in a browser on the Jetson.

---

## MLC Benchmarks

### 1. Start the MLC container

```bash
jetson-containers run \
    -v /path/to/mlc_models:/mlc_models \
    dustynv/mlc:0.1.4-r36.4.2
```

### 2. Convert models manually

Models must be converted to MLC format before benchmarking.
Run the following steps inside the MLC container.

Clone the model from HuggingFace:

```bash
# Example: Llama-3.1-8B-Instruct
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

Convert the weights:

```bash
# General form
mlc_llm convert_weight <path-to-model> \
    --quantization <quantization-type> \
    -o <output-path>

# Example: Llama-3.1-8B-Instruct in Q0F16
mlc_llm convert_weight Llama-3.1-8B-Instruct \
    --quantization q0f16 \
    -o /mlc_models/Llama-3.1-8B-Instruct-q0f16-MLC
```

Generate the model config:

```bash
# General form
mlc_llm gen_config <path-to-model> \
    --quantization <quantization-type> \
    --conv-template <model-template> \
    -o <output-path>

# Example: Llama-3.1-8B-Instruct in Q0F16
mlc_llm gen_config Llama-3.1-8B-Instruct \
    --quantization q0f16 \
    --conv-template llama-3 \
    -o /mlc_models/Llama-3.1-8B-Instruct-q0f16-MLC
```

### 3. Run the benchmarks

The MLC benchmark runner script is based on the original
`benchmark.sh` from the jetson-containers repository at `jetson-containers/packages/llm/mlc`. You can either
adapt that script directly or create your own based on `mlc_bench_runner.sh`
in your repository. The key modifications made relative to the original are:

- `MAX_NUM_PROMPTS` reduced from 4 to 1
- `PROMPT` changed from the 16-token to the 512-token completion file
- `SKIP_QUANTIZATION` set to `yes` to use pre-converted local models
- Model path changed to use local `/mlc_models` directory

Edit `mlc_bench_runner.sh` and set the volume mount path for your
mlc_models directory, then run from the host:

```bash
bash mlc_bench_runner.sh
```

The 512-token prompt file used for benchmarking is included in the
jetson-containers repository at:
`jetson-containers/data/prompts/completion_512.json`

> **Note:** The MLC benchmark runs each configuration once with no
> repetition averaging. Standard deviation is not reported.
> If measurement stability is required, runs must be repeated manually
> and standard deviation calculated accordingly.

---

## Benchmark Configuration

All benchmarks in this repository used the following configuration:

| Parameter | Value |
|---|---|
| Prompt tokens | 512 |
| Generated tokens | 128 |
| Batch size | 2048 |
| KV cache precision | F16 |
| GPU layer offload | All layers (`-ngl 99`) |
| Flash attention | Enabled (`-fa 1`, llama.cpp only) |
| Repetitions | 5 (llama.cpp) / 1 (MLC) |
| Power mode | MAXN (`nvpmodel -m 0`) |

---

## Output

### llama.cpp

Results are written to a CSV file with the following columns:

```
timestamp, model, pp, tg, prefill_rate, decode_rate, prefill_stddev, decode_stddev, memory in MB
```

Example row:

```
16/04/2026 01:41:39, meta-llama-3.1-8b-instruct.f16.gguf, 512, 128, 177.864472, 3.399341, 0.017416, 0.000358, 16013.9765625
```

`pp` is prompt tokens processed (prefill), `tg` is tokens generated (decode).
Rates are in tokens/second. Memory is peak usage in MB via `ru_maxrss`.

### MLC

Results are written to a CSV file with the following columns:

```
timestamp, hostname, api, model, precision, input_tokens, output_tokens, prefill_time, prefill_rate, decode_time, decode_rate, memory
```

Example row:

```
20260219 06:42:06, ubuntu, mlc, /mlc_models/Llama-3.1-8B-Instruct-q0f16-MLC, MLC, 495, 128, 0.9703276990000002, 510.1369367381111, 11.345863208314961, 11.281644917611342, 34704.77734375
```

Rates are in tokens/second. Memory is peak usage in MB via `ru_maxrss`.
Note that MLC processed 495 prompt tokens instead of 512 due to tokenizer differences.
