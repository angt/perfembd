use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::{error, info};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter},
    path::PathBuf,
    time::{Duration, Instant},
};
use thiserror::Error;

const CORPUS: &str = include_str!("../corpus.txt");

/// Benchmark tool for embedding endpoints
#[derive(Parser, Debug)]
struct Args {
    /// Target API base URL.
    #[arg(default_value = "http://localhost:3000/v1")]
    target_url: String,

    /// Path to the corpus file (one document per line).
    #[arg(short, long)]
    corpus_file: Option<PathBuf>,

    /// Number of documents per request batch.
    #[arg(short, long, default_value_t = 128)]
    batch_size: usize,

    /// Number of benchmark iterations (full passes over the corpus).
    #[arg(short = 'n', long, default_value_t = 20)]
    iterations: usize,

    /// Output file for results (JSON format).
    #[arg(short = 'o', long, default_value = "out.json")]
    output_file: PathBuf,

    /// Optional Bearer authentication token for the API.
    #[arg(long)]
    token: Option<String>,

    /// Optional: Embedding model name.
    #[arg(short = 'm', long, default_value = "none")]
    model_name: String,
}

#[derive(Serialize)]
struct EmbeddingRequest {
    input: Vec<String>,
    model: String,
}

#[derive(Deserialize, Debug)]
struct EmbeddingResponse {
    usage: UsageData,
}

#[derive(Deserialize, Debug)]
struct UsageData {
    total_tokens: u64,
}

pub struct Stats {
    min: Option<u32>,
    max: Option<u32>,
    sum: u64,
    count: u64,
    lower_half: BinaryHeap<u32>,
    upper_half: BinaryHeap<Reverse<u32>>,
}

impl Stats {
    pub fn new() -> Self {
        Stats {
            min: None,
            max: None,
            sum: 0,
            count: 0,
            lower_half: BinaryHeap::new(),
            upper_half: BinaryHeap::new(),
        }
    }

    pub fn update(&mut self, value: u32) {
        self.min = Some(self.min.map(|m| m.min(value)).unwrap_or(value));
        self.max = Some(self.max.map(|m| m.max(value)).unwrap_or(value));

        self.sum += value as u64;
        self.count += 1;

        if self.lower_half.peek().map_or(true, |&m| value <= m) {
            self.lower_half.push(value);
        } else {
            self.upper_half.push(Reverse(value));
        }
        if self.lower_half.len() > self.upper_half.len() + 1 {
            if let Some(v) = self.lower_half.pop() {
                self.upper_half.push(Reverse(v));
            }
        } else if self.upper_half.len() > self.lower_half.len() {
            if let Some(Reverse(v)) = self.upper_half.pop() {
                self.lower_half.push(v);
            }
        }
    }

    pub fn min(&self) -> Option<u32> {
        self.min
    }

    pub fn max(&self) -> Option<u32> {
        self.max
    }

    pub fn avg(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum as f64 / self.count as f64)
        }
    }

    pub fn median(&self) -> Option<f64> {
        if self.lower_half.len() == self.upper_half.len() {
            match (self.lower_half.peek(), self.upper_half.peek()) {
                (Some(&l), Some(&Reverse(r))) => Some((l as f64 + r as f64) / 2.0),
                _ => None,
            }
        } else {
            self.lower_half.peek().map(|&val| val as f64)
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }
}

struct BenchProgress {
    count: ProgressBar,
    errors: ProgressBar,
    min: ProgressBar,
    max: ProgressBar,
    avg: ProgressBar,
    median: ProgressBar,
    iter: ProgressBar,
}

impl BenchProgress {
    fn new(m: &MultiProgress, count: usize) -> Result<BenchProgress> {
        let stats_style = ProgressStyle::with_template("{prefix:>10}: {msg}")?;

        let iter_style = ProgressStyle::with_template(
            "{spinner:.green} Bench {pos:>4}/{len} [{elapsed_precise}] [{bar:40.cyan/blue}] {percent:>3}% | ETA: {eta}"
        )?.progress_chars("#>-");

        let make_stat_bar = |prefix: &'static str, msg: &'static str| {
            let pb = m.add(ProgressBar::new(1));
            pb.set_style(stats_style.clone());
            pb.set_prefix(prefix);
            pb.set_message(msg);
            pb
        };

        Ok(BenchProgress {
            count: make_stat_bar("Successful requests", "0"),
            errors: make_stat_bar("Failed requests", "0"),
            min: make_stat_bar("min tokens/s", "N/A"),
            max: make_stat_bar("max tokens/s", "N/A"),
            avg: make_stat_bar("avg tokens/s", "N/A"),
            median: make_stat_bar("median tokens/s", "N/A"),
            iter: {
                let pb = m.add(ProgressBar::new(count as u64));
                pb.set_style(iter_style);
                pb
            },
        })
    }

    fn update(&self, stats: &Stats, errors: u64) {
        self.count.set_message(format!("{}", stats.count()));
        self.errors.set_message(format!("{}", errors));
        self.min.set_message(stats.min()
            .map_or("N/A".into(), |v| format!("{}", v)));
        self.max.set_message(stats.max()
            .map_or("N/A".into(), |v| format!("{}", v)));
        self.avg.set_message(stats.avg()
            .map_or("N/A".into(), |v| format!("{:.1}", v)));
        self.median.set_message(stats.median()
            .map_or("N/A".into(), |v| format!("{:.1}", v)));
        self.iter.inc(1);
    }

    fn clear(&self) {
        self.count.finish();
        self.errors.finish();
        self.min.finish();
        self.max.finish();
        self.avg.finish();
        self.median.finish();
        self.iter.finish_and_clear();
    }
}

struct WarmupProgress {
    iter: ProgressBar,
}

impl WarmupProgress {
    fn new(m: &MultiProgress, count: usize) -> Result<Self> {
        let iter_style = ProgressStyle::with_template(
            "{spinner:.yellow} Warm-up {pos:>4}/{len} [{elapsed_precise}] [{bar:40.yellow/orange}] {percent:>3}% | ETA: {eta}"
        )?.progress_chars("#>-");

        Ok(WarmupProgress {
            iter: {
                let pb = m.add(ProgressBar::new(count as u64));
                pb.set_style(iter_style);
                pb
            },
        })
    }

    fn update(&self) {
        self.iter.inc(1);
    }

    fn clear(&self) {
        self.iter.finish_and_clear();
    }
}

fn read_corpus<R: BufRead>(reader: R) -> Result<Vec<String>> {
    reader
        .lines()
        .map(|line| {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        })
        .filter_map(Result::transpose)
        .collect()
}

fn build_client(token: Option<&str>) -> Result<Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    if let Some(token) = token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", token)
                .parse()
                .context("Failed to parse Authorization header")?,
        );
    }
    Client::builder()
        .user_agent(format!("{}/{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION")))
        .default_headers(headers)
        .use_rustls_tls()
        .pool_max_idle_per_host(10)
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(90))
        .build()
        .context("Failed to build HTTP client")
}

#[derive(Error, Debug)]
pub enum BatchError {
    #[error("API request failed with status {status}: {source}")]
    ApiError { status: reqwest::StatusCode, source: reqwest::Error },
    #[error("API request error: {0}")]
    RequestError(#[from] reqwest::Error),
    #[error("Failed to compute token/s")]
    Invalid,
}

pub async fn run_batch(
    client: &reqwest::Client,
    url: &str,
    batch: &[String],
    model_name: &str,
) -> Result<u32, BatchError> {
    let request_payload = EmbeddingRequest {
        input: batch.to_vec(),
        model: model_name.to_string(),
    };
    let start_time = Instant::now();
    let response = client.post(url).json(&request_payload).send().await?;
    let duration = start_time.elapsed().as_secs_f64();
    let status = response.status();

    if !status.is_success() {
        let source = response.error_for_status().unwrap_err();
        return Err(BatchError::ApiError { status, source });
    }
    let result: EmbeddingResponse = response.json().await?;
    let tokens = result.usage.total_tokens;

    if duration <= 0.0 {
        return Err(BatchError::Invalid);
    }
    let tps_f64 = tokens as f64 / duration;

    if !tps_f64.is_finite() {
        return Err(BatchError::Invalid);
    }
    let tps = tps_f64.round();

    if tps <= 0.0 || tps > u32::MAX as f64 {
        return Err(BatchError::Invalid);
    }
    Ok(tps as u32)
}

#[derive(Serialize)]
struct BenchResult {
    url: String,
    model_name: String,
    batch_size: usize,
    iterations: usize,
    successful_requests: u64,
    errors: u64,
    min_tokens_per_second: Option<u32>,
    max_tokens_per_second: Option<u32>,
    avg_tokens_per_second: Option<f64>,
    median_tokens_per_second: Option<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::new()
        .format_timestamp(None)
        .format_module_path(false)
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = Args::parse();
    let url = format!("{}/embeddings", args.target_url.trim_end_matches('/'));

    let documents = if let Some(path) = &args.corpus_file {
        let file =
            File::open(path).with_context(|| format!("Failed to open corpus: {:?}", path))?;
        read_corpus(BufReader::new(file))
    } else {
        read_corpus(BufReader::new(CORPUS.as_bytes()))
    }
    .with_context(|| format!("Failed to parse corpus"))?;

    let batches: Vec<_> = documents.chunks(args.batch_size).map(Vec::from).collect();

    let client = build_client(args.token.as_deref())?;
    let m = MultiProgress::with_draw_target(ProgressDrawTarget::stdout());

    let warmup_passes = 5;
    let warmup = WarmupProgress::new(&m, warmup_passes * batches.len())?;

    for _ in 0..warmup_passes {
        for batch in &batches {
            if let Err(e) = run_batch(&client, &url, &batch, &args.model_name).await {
                bail!("Warm-up failed: {}", e);
            }
            warmup.update();
        }
    }
    warmup.clear();

    let bench = BenchProgress::new(&m, args.iterations * batches.len())?;
    let mut stats = Stats::new();
    let mut errors: u64 = 0;

    for _ in 0..args.iterations {
        for batch in &batches {
            match run_batch(&client, &url, &batch, &args.model_name).await {
                Ok(tps) => stats.update(tps),
                Err(_err) => errors += 1,
            }
            bench.update(&stats, errors);
        }
    }
    bench.clear();

    let results = BenchResult {
        url,
        model_name: args.model_name,
        batch_size: args.batch_size,
        iterations: args.iterations,
        successful_requests: stats.count(),
        errors: errors,
        min_tokens_per_second: stats.min(),
        max_tokens_per_second: stats.max(),
        avg_tokens_per_second: stats.avg(),
        median_tokens_per_second: stats.median(),
    };
    let file = File::create(&args.output_file)
        .with_context(|| format!("Failed to create: {}", args.output_file.display()))?;

    serde_json::to_writer_pretty(BufWriter::new(file), &results)
        .with_context(|| format!("Failed to write: {}", args.output_file.display()))?;

    info!("Results written in {}", args.output_file.display());

    Ok(())
}
