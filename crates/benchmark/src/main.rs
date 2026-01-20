//! ESICUP Benchmark Runner CLI

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use u_nesting_benchmark::{BenchmarkConfig, BenchmarkRunner, DatasetParser};
use u_nesting_core::Strategy;

#[derive(Parser)]
#[command(name = "bench-runner")]
#[command(about = "ESICUP Benchmark Runner for U-Nesting")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available ESICUP datasets
    List,

    /// Run benchmark on a single dataset
    Run {
        /// Dataset name (e.g., SHAPES, SHIRTS, SWIM)
        #[arg(short, long)]
        dataset: String,

        /// Instance name within the dataset (e.g., shapes0, shapes1)
        #[arg(short, long)]
        instance: Option<String>,

        /// Strategies to benchmark
        #[arg(short, long, value_enum, default_values_t = vec![StrategyArg::Blf, StrategyArg::Nfp])]
        strategies: Vec<StrategyArg>,

        /// Time limit per run in seconds
        #[arg(short, long, default_value = "60")]
        time_limit: u64,

        /// Number of runs per configuration
        #[arg(short, long, default_value = "1")]
        runs: usize,

        /// Output file for results (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output file for CSV results
        #[arg(long)]
        csv: Option<PathBuf>,
    },

    /// Run benchmarks on multiple datasets
    RunAll {
        /// Preset configuration
        #[arg(short, long, value_enum, default_value = "quick")]
        preset: Preset,

        /// Output file for results (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output file for CSV results
        #[arg(long)]
        csv: Option<PathBuf>,
    },

    /// Run benchmark from a local JSON file
    RunFile {
        /// Path to the JSON dataset file
        file: PathBuf,

        /// Strategies to benchmark
        #[arg(short, long, value_enum, default_values_t = vec![StrategyArg::Blf, StrategyArg::Nfp])]
        strategies: Vec<StrategyArg>,

        /// Time limit per run in seconds
        #[arg(short, long, default_value = "60")]
        time_limit: u64,

        /// Output file for results (JSON)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Download a dataset
    Download {
        /// Dataset name
        dataset: String,

        /// Output directory
        #[arg(short, long, default_value = "datasets")]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum StrategyArg {
    /// Bottom-Left Fill
    Blf,
    /// NFP-guided placement
    Nfp,
    /// Genetic Algorithm
    Ga,
    /// BRKGA
    Brkga,
    /// Simulated Annealing
    Sa,
}

impl From<StrategyArg> for Strategy {
    fn from(arg: StrategyArg) -> Self {
        match arg {
            StrategyArg::Blf => Strategy::BottomLeftFill,
            StrategyArg::Nfp => Strategy::NfpGuided,
            StrategyArg::Ga => Strategy::GeneticAlgorithm,
            StrategyArg::Brkga => Strategy::Brkga,
            StrategyArg::Sa => Strategy::SimulatedAnnealing,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    /// Quick benchmarks (BLF, NFP, 5s timeout)
    Quick,
    /// Standard benchmarks (all strategies, 60s timeout)
    Standard,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::List => {
            println!("Available ESICUP Datasets:");
            println!("==========================");
            for name in DatasetParser::list_available_datasets() {
                println!("  - {}", name);
            }
            println!("\nUse 'bench-runner run -d <DATASET>' to run benchmarks");
        }

        Commands::Run {
            dataset,
            instance,
            strategies,
            time_limit,
            runs,
            output,
            csv,
        } => {
            let parser = DatasetParser::new();

            // Determine instance name
            let instance_name = instance.unwrap_or_else(|| dataset.to_lowercase());

            println!("Downloading dataset: {}/{}", dataset, instance_name);
            let ds = parser.download_and_parse(&dataset, &instance_name)?;

            let strategies: Vec<Strategy> = strategies.into_iter().map(Into::into).collect();

            let config = BenchmarkConfig::new()
                .with_strategies(strategies)
                .with_time_limit(time_limit * 1000)
                .with_runs_per_config(runs);

            let runner = BenchmarkRunner::new(config);
            let results = runner.run_dataset(&ds);

            results.print_summary();

            if let Some(path) = output {
                results.save_json(&path)?;
                println!("Results saved to: {}", path.display());
            }

            if let Some(path) = csv {
                results.save_csv(&path)?;
                println!("CSV saved to: {}", path.display());
            }
        }

        Commands::RunAll { preset, output, csv } => {
            let config = match preset {
                Preset::Quick => BenchmarkConfig::quick(),
                Preset::Standard => BenchmarkConfig::standard(),
            };

            let parser = DatasetParser::new();
            let runner = BenchmarkRunner::new(config);

            // Define a set of well-known instances
            let instances = [
                ("SHAPES", "shapes0"),
                ("SHAPES", "shapes1"),
                ("SHIRTS", "shirts"),
                ("SWIM", "swim"),
            ];

            let mut all_results = u_nesting_benchmark::BenchmarkResult::new();

            for (dataset, instance) in &instances {
                println!("\nDownloading {}/{}...", dataset, instance);
                match parser.download_and_parse(dataset, instance) {
                    Ok(ds) => {
                        let results = runner.run_dataset(&ds);
                        for run in results.runs {
                            all_results.add_run(run);
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to download {}/{}: {}", dataset, instance, e);
                    }
                }
            }

            all_results.print_summary();

            // Print strategy comparison
            println!("\nStrategy Comparison:");
            println!("{:-<60}", "");
            for summary in all_results.summary_by_strategy() {
                println!(
                    "  {:<20} runs={:<3} avg_util={:.1}% avg_time={}ms",
                    summary.strategy,
                    summary.run_count,
                    summary.avg_utilization * 100.0,
                    summary.avg_time_ms
                );
            }

            if let Some(path) = output {
                all_results.save_json(&path)?;
                println!("\nResults saved to: {}", path.display());
            }

            if let Some(path) = csv {
                all_results.save_csv(&path)?;
                println!("CSV saved to: {}", path.display());
            }
        }

        Commands::RunFile {
            file,
            strategies,
            time_limit,
            output,
        } => {
            let parser = DatasetParser::new();
            let ds = parser.parse_file(&file)?;

            let strategies: Vec<Strategy> = strategies.into_iter().map(Into::into).collect();

            let config = BenchmarkConfig::new()
                .with_strategies(strategies)
                .with_time_limit(time_limit * 1000);

            let runner = BenchmarkRunner::new(config);
            let results = runner.run_dataset(&ds);

            results.print_summary();

            if let Some(path) = output {
                results.save_json(&path)?;
                println!("Results saved to: {}", path.display());
            }
        }

        Commands::Download { dataset, output } => {
            let parser = DatasetParser::new();
            let instance = dataset.to_lowercase();

            println!("Downloading {}/{}...", dataset, instance);
            let ds = parser.download_and_parse(&dataset, &instance)?;

            std::fs::create_dir_all(&output)?;
            let file_path = output.join(format!("{}.json", instance));
            let json = serde_json::to_string_pretty(&ds)?;
            std::fs::write(&file_path, json)?;

            println!("Dataset saved to: {}", file_path.display());
            println!("  Items: {}", ds.items.len());
            println!("  Total pieces: {}", ds.expand_items().len());
            println!("  Strip height: {}", ds.strip_height);
        }
    }

    Ok(())
}
