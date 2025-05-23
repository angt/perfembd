# perfembd

A command-line tool for benchmarking text embedding API endpoints.

## Install

There are two ways to install `perfembd`:

### From Precompiled Binaries

Binaries are provided for Linux and macOS (`x86_64` and `aarch64`).

    OS=linux     # or macos
    ARCH=x86_64  # or aarch64
    V=0.4.1
    curl -sSf https://github.com/angt/perfembd/releases/download/v$V/perfembd-$ARCH-$OS.gz | gunzip > perfembd
    chmod +x perfembd

### From Source (Requires Rust toolchain)

    cargo install --git https://github.com/angt/perfembd

### Args

    $ ./perfembd --help
    Benchmark tool for embedding endpoints

    Usage: perfembd [OPTIONS] [URL]

    Arguments:
      [URL]  Target API base URL [default: http://localhost:3000/v1]

    Options:
      -c, --corpus <CORPUS>
              Path to the corpus file (one document per line)
      -b, --batch-size <BATCH_SIZE>
              Number of documents per request batch [default: 128]
      -n, --iterations <ITERATIONS>
              Number of benchmark iterations (full passes over the corpus) [default: 256]
      -w, --warmup-iterations <WARMUP_ITERATIONS>
              Number of warm-up iterations (full passes over the corpus) [default: 24]
      -o, --output-file <OUTPUT_FILE>
              Output file for results (JSON format) [default: out.json]
          --token <TOKEN>
              Optional Bearer authentication token for the API
      -m, --model-name <MODEL_NAME>
              Optional: Embedding model name [default: none]
          --connect-timeout <CONNECT_TIMEOUT>
              HTTP connect timeout in seconds [default: 15]
          --request-timeout <REQUEST_TIMEOUT>
              HTTP request timeout in seconds (for the entire request lifecycle) [default: 90]
          --pool-idle-timeout <POOL_IDLE_TIMEOUT>
              HTTP pool idle timeout in seconds (how long an idle connection is kept alive) [default: 90]
          --insecure
              Accept invalid TLS/SSL certificates
      -h, --help
              Print help

