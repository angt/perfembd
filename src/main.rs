use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use log::error;
use rand::prelude::IndexedRandom;
use rand::Rng;
use reqwest::Client;
use serde::{Deserialize, Serialize};
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
    url: String,

    /// Path to the corpus file (one document per line).
    #[arg(short = 'c', long)]
    corpus: Option<PathBuf>,

    /// Number of documents per request batch.
    #[arg(short = 'b', long, default_value_t = 128)]
    batch_size: usize,

    /// Number of benchmark iterations (full passes over the corpus).
    #[arg(short = 'n', long, default_value_t = 256)]
    iterations: usize,

    /// Number of warm-up iterations (full passes over the corpus).
    #[arg(short = 'w', long, default_value_t = 24)]
    warmup_iterations: usize,

    /// Output file for results (JSON format).
    #[arg(short = 'o', long, default_value = "out.json")]
    output_file: PathBuf,

    /// Optional Bearer authentication token for the API.
    #[arg(long)]
    token: Option<String>,

    /// Optional: Embedding model name.
    #[arg(short = 'm', long, default_value = "none")]
    model_name: String,

    /// HTTP connect timeout in seconds.
    #[arg(long, default_value_t = 15)]
    connect_timeout: u64,

    /// HTTP request timeout in seconds (for the entire request lifecycle).
    #[arg(long, default_value_t = 90)]
    request_timeout: u64,

    /// HTTP pool idle timeout in seconds (how long an idle connection is kept alive).
    #[arg(long, default_value_t = 90)]
    pool_idle_timeout: u64,

    /// Accept invalid TLS/SSL certificates.
    #[arg(long, default_value_t = false)]
    insecure: bool,
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

#[derive(Clone, Debug, Default)]
pub struct Bucket {
    min: Option<u32>,
    max: Option<u32>,
    sum: u64,
    sum_sq: u128,
    count: u64,
    values: Vec<u32>,
}

impl Bucket {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn update(&mut self, value: u32) {
        self.min = Some(self.min.map(|m| m.min(value)).unwrap_or(value));
        self.max = Some(self.max.map(|m| m.max(value)).unwrap_or(value));

        self.sum += value as u64;
        self.sum_sq += (value as u128) * (value as u128);
        self.count += 1;

        let pos = match self.values.binary_search(&value) {
            Ok(index) => index,
            Err(index) => index,
        };
        self.values.insert(pos, value);
    }

    pub fn min(&self) -> Option<u32> {
        self.min
    }

    pub fn max(&self) -> Option<u32> {
        self.max
    }

    pub fn avg(&self) -> Option<u32> {
        if self.count == 0 {
            None
        } else {
            Some((self.sum / self.count) as u32)
        }
    }

    fn p(&self, percent: usize) -> Option<u32> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.values[(self.values.len() - 1) * percent / 100])
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn variance(&self) -> Option<u32> {
        if self.count < 2 {
            return None;
        }
        let mean_sq = self.sum_sq / self.count as u128;
        let mean = (self.sum / self.count) as i128;
        let variance = mean_sq as i128 - (mean * mean);

        if variance < 0 {
            Some(0)
        } else {
            Some(variance as u32)
        }
    }

    pub fn stdev(&self) -> Option<u32> {
        self.variance().map(|v| v.isqrt())
    }
}

#[derive(Debug)]
pub struct Corpus {
    docs: Vec<String>,
    batch_size: usize,
    rng: rand::rngs::ThreadRng,
}

impl Corpus {
    pub fn new(path: &Option<PathBuf>, batch_size: usize) -> Result<Self> {
        let docs = if let Some(path) = path {
            let file =
                File::open(path).with_context(|| format!("Failed to open corpus: {:?}", path))?;
            Self::read_corpus(BufReader::new(file))
        } else {
            Self::read_corpus(BufReader::new(CORPUS.as_bytes()))
        }
        .with_context(|| format!("Failed to parse corpus"))?;

        if docs.is_empty() {
            bail!("Loaded corpus is empty. Cannot proceed with benchmarking.");
        }
        Ok(Self {
            docs,
            batch_size,
            rng: rand::rng(),
        })
    }

    pub fn random_batch(&mut self) -> Vec<String> {
        let n = self.rng.random_range(1..=self.batch_size);

        self.docs
            .choose_multiple(&mut self.rng, n)
            .cloned()
            .collect()
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
}

struct BucketProgress {
    pb: ProgressBar,
}

impl BucketProgress {
    fn new(m: &MultiProgress, prefix: String) -> Result<Self> {
        let style = ProgressStyle::with_template("{prefix:>7}: {msg}")?;
        let pb = m.add(ProgressBar::new(1));
        pb.set_style(style.clone());
        pb.set_prefix(prefix);
        pb.set_message(Self::format(
            String::from(""),
            String::from(""),
            String::from(""),
            String::from(""),
            String::from(""),
            String::from(""),
            String::from(""),
            String::from(""),
        ));
        Ok(Self { pb })
    }

    fn format(
        count: String,
        min: String,
        p25: String,
        p50: String,
        p75: String,
        max: String,
        avg: String,
        stdev: String,
    ) -> String {
        format!(
            " {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} | {:>7} ",
            count, min, p25, p50, p75, max, avg, stdev
        )
    }

    fn header(&self) {
        let message = Self::format(
            String::from("count"),
            String::from("min"),
            String::from("p25"),
            String::from("median"),
            String::from("p75"),
            String::from("max"),
            String::from("avg"),
            String::from("stdev"),
        );
        self.pb.set_message(message);
    }

    fn update(&self, bucket: &Bucket) {
        let message = Self::format(
            bucket.count().to_string(),
            bucket.min().map(|v| v.to_string()).unwrap_or_default(),
            bucket.p(25).map(|v| v.to_string()).unwrap_or_default(),
            bucket.p(50).map(|v| v.to_string()).unwrap_or_default(),
            bucket.p(75).map(|v| v.to_string()).unwrap_or_default(),
            bucket.max().map(|v| v.to_string()).unwrap_or_default(),
            bucket.avg().map(|v| v.to_string()).unwrap_or_default(),
            bucket.stdev().map(|v| v.to_string()).unwrap_or_default(),
        );
        self.pb.set_message(message);
    }

    fn clear(&self) {
        self.pb.finish();
    }
}

struct BenchProgress {
    header: BucketProgress,
    buckets: Vec<BucketProgress>,
    pb: ProgressBar,
}

impl BenchProgress {
    fn new(m: &MultiProgress, size: usize, count: usize) -> Result<Self> {
        let header = BucketProgress::new(m, "tokens".into())?;
        header.header();

        let mut buckets = Vec::with_capacity(size);

        for i in 0..size {
            buckets.push(BucketProgress::new(m, 2usize.pow(i as u32).to_string())?);
        }
        let pb = m.add(ProgressBar::new(count as u64));
        pb.set_style(make_progress_style("Bench", "green", "cyan/blue"));

        Ok(Self {
            header,
            buckets,
            pb,
        })
    }

    fn update(&self, buckets: &[Bucket], _errors: u64) {
        if self.buckets.len() == buckets.len() {
            for (s, new) in self.buckets.iter().zip(buckets) {
                s.update(new);
            }
        }
        self.pb.inc(1);
    }

    fn clear(&self) {
        self.header.clear();

        for bucket in &self.buckets {
            bucket.clear();
        }
        self.pb.finish_and_clear();
    }
}

fn make_progress_style(title: &str, spinner_color: &str, bar_colors: &str) -> ProgressStyle {
    let template = format!(
        "{{spinner:.{spinner_color}}} {title} {{pos:>4}}/{{len}} \
         [{{elapsed_precise}}] [{{bar:40.{bar_colors}}}] \
         {{percent:>3}}% | ETA: {{eta}}",
    );
    ProgressStyle::with_template(&template)
        .expect("invalid template")
        .progress_chars("#>-")
}

struct WarmupProgress {
    pb: ProgressBar,
}

impl WarmupProgress {
    fn new(m: &MultiProgress, count: usize) -> Result<Self> {
        let pb = m.add(ProgressBar::new(count as u64));
        pb.set_style(make_progress_style("Warm-up", "yellow", "yellow/orange"));
        Ok(Self { pb })
    }

    fn update(&self) {
        self.pb.inc(1);
    }

    fn clear(&self) {
        self.pb.finish_and_clear();
    }
}

fn build_client(args: &Args) -> Result<Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(
        reqwest::header::CONTENT_TYPE,
        "application/json".parse().unwrap(),
    );
    if let Some(token) = &args.token {
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", token)
                .parse()
                .context("Failed to parse Authorization header")?,
        );
    }
    Client::builder()
        .user_agent(format!(
            "{}/{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION")
        ))
        .default_headers(headers)
        .use_rustls_tls()
        .pool_idle_timeout(Duration::from_secs(args.pool_idle_timeout))
        .connect_timeout(Duration::from_secs(args.connect_timeout))
        .timeout(Duration::from_secs(args.request_timeout))
        .danger_accept_invalid_certs(args.insecure)
        .build()
        .context("Failed to build HTTP client")
}

#[derive(Error, Debug)]
pub enum BatchError {
    #[error("API request failed with status {status}: {source}")]
    ApiError {
        status: reqwest::StatusCode,
        source: reqwest::Error,
    },
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
) -> Result<(u32, u64), BatchError> {
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
    Ok((tps as u32, tokens))
}

#[derive(Serialize)]
struct StatResult {
    tokens: usize,
    requests_in_bucket: u64,
    min_tokens_per_second: Option<u32>,
    p25_tokens_per_second: Option<u32>,
    p50_tokens_per_second: Option<u32>,
    p75_tokens_per_second: Option<u32>,
    max_tokens_per_second: Option<u32>,
    avg_tokens_per_second: Option<u32>,
    stdev_tokens_per_second: Option<u32>,
}

#[derive(Serialize)]
struct BenchResult {
    url: String,
    model_name: String,
    iterations: usize,
    errors: u64,
    stats: Vec<StatResult>,
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
    let url = format!("{}/embeddings", args.url.trim_end_matches('/'));

    let mut corpus = Corpus::new(&args.corpus, args.batch_size)?;
    let client = build_client(&args)?;
    let m = MultiProgress::with_draw_target(ProgressDrawTarget::stdout());

    let warmup = WarmupProgress::new(&m, args.warmup_iterations)?;

    for _ in 0..args.warmup_iterations {
        let batch = corpus.random_batch();

        if let Err(e) = run_batch(&client, &url, &batch, &args.model_name).await {
            bail!("Warm-up failed: {}", e);
        }
        warmup.update();
    }
    warmup.clear();

    let mut buckets: Vec<Bucket> = vec![Bucket::new(); 15];
    let mut errors: u64 = 0;
    let bench = BenchProgress::new(&m, buckets.len(), args.iterations)?;

    for _ in 0..args.iterations {
        let batch = corpus.random_batch();

        match run_batch(&client, &url, &batch, &args.model_name).await {
            Ok((tps, tokens)) => {
                let idx = (tokens.ilog2() as usize).min(buckets.len() - 1);
                buckets[idx].update(tps);
            }
            Err(_err) => errors += 1,
        }
        bench.update(&buckets, errors);
    }
    bench.clear();

    let mut stats: Vec<StatResult> = Vec::new();

    for (idx, stat) in buckets.iter().enumerate() {
        if stat.count() > 0 {
            let result = StatResult {
                tokens: 2usize.pow(idx as u32),
                requests_in_bucket: stat.count(),
                min_tokens_per_second: stat.min(),
                p25_tokens_per_second: stat.p(25),
                p50_tokens_per_second: stat.p(50),
                p75_tokens_per_second: stat.p(75),
                max_tokens_per_second: stat.max(),
                avg_tokens_per_second: stat.avg(),
                stdev_tokens_per_second: stat.stdev(),
            };
            stats.push(result);
        }
    }
    let bench = BenchResult {
        url,
        model_name: args.model_name,
        iterations: args.iterations,
        errors,
        stats,
    };
    let file = File::create(&args.output_file)
        .with_context(|| format!("Failed to create: {}", args.output_file.display()))?;

    serde_json::to_writer_pretty(BufWriter::new(file), &bench)
        .with_context(|| format!("Failed to write: {}", args.output_file.display()))?;

    Ok(())
}
