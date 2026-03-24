//! Thread-per-core multi-runtime telemetry example.
//!
//! Demonstrates `TracedRuntimeGroup` with a thread-per-core I/O architecture.
//! N worker runtimes each process simulated I/O independently, with a ring of
//! channels between workers to create cross-runtime wake events visible in the
//! trace viewer.
//!
//! What to look for in the trace viewer:
//! - Each `io-workers-N` lane shows poll spans from concurrent tasks
//! - Wake events cross between workers when channel messages arrive
//!   (click a task to see wake arrows)
//! - Worker 0 has occasional long polls (simulated CPU-bound work) that
//!   delay tasks waiting on it — visible as scheduling gaps
//! - Queue depth varies per worker based on load
//!
//! Run:
//!   cargo run --release --example multi_runtime
//!
//! View:
//!   python3 dial9-tokio-telemetry/serve.py
//!   # open http://localhost:8000, drag in /tmp/dial9-multi-runtime-trace.bin

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use dial9_tokio_telemetry::telemetry::{RotatingWriter, TracedRuntimeGroup};

const NUM_WORKERS: usize = 4;
const DURATION_SECS: u64 = 5;
const TRACE_PATH: &str = "/tmp/dial9-multi-runtime-trace.bin";

fn current_thread_builder() -> tokio::runtime::Builder {
    let mut b = tokio::runtime::Builder::new_current_thread();
    b.enable_all();
    b
}

/// Simulate an I/O operation with variable latency.
async fn simulate_io(id: u64) {
    let delay = Duration::from_micros(200 + (id % 300) * 5);
    tokio::time::sleep(delay).await;
}

/// Simulate occasional CPU-heavy work (e.g., checksum computation) that
/// blocks the event loop and delays other tasks on this worker.
fn burn_cpu(micros: u64) {
    let start = Instant::now();
    let mut x = 0u64;
    while start.elapsed() < Duration::from_micros(micros) {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    std::hint::black_box(x);
}

fn main() -> std::io::Result<()> {
    let writer = RotatingWriter::single_file(TRACE_PATH)?;

    let mut group = TracedRuntimeGroup::builder()
        .with_task_tracking(true)
        .build(writer)?;

    let workers = group.register_group(
        "io-workers",
        (0..NUM_WORKERS).map(|_| current_thread_builder()),
    )?;

    let guard = group.start();
    let handle = guard.handle();
    let start = Instant::now();
    let deadline = start + Duration::from_secs(DURATION_SECS);
    let completed = Arc::new(AtomicU64::new(0));

    // Cross-worker handoff channels: worker i sends to worker (i+1)%N
    // after completing work, creating cross-runtime wake events.
    let mut handoff_txs = Vec::new();
    let mut handoff_rxs = Vec::new();
    for _ in 0..NUM_WORKERS {
        let (tx, rx) = tokio::sync::mpsc::channel::<u64>(64);
        handoff_txs.push(tx);
        handoff_rxs.push(rx);
    }

    std::thread::scope(|s| {
        for (i, rt) in workers.iter().enumerate() {
            let worker_handle = handle.clone();
            let completed = completed.clone();
            let deadline = deadline;
            // This worker sends handoffs to the next worker
            let handoff_tx = handoff_txs[(i + 1) % NUM_WORKERS].clone();
            // This worker receives handoffs from the previous worker
            let mut handoff_rx = handoff_rxs.remove(0);

            s.spawn(move || {
                rt.block_on(async {
                    let mut request_id = i as u64;

                    loop {
                        if Instant::now() >= deadline {
                            break;
                        }

                        // Do I/O work, then hand off to the next worker
                        let completed_clone = completed.clone();
                        let handoff_tx_clone = handoff_tx.clone();
                        worker_handle.spawn(async move {
                            simulate_io(request_id).await;
                            let _ = handoff_tx_clone.send(request_id).await;
                            completed_clone.fetch_add(1, Ordering::Relaxed);
                        });

                        // Process handoffs arriving from the previous worker
                        while let Ok(id) = handoff_rx.try_recv() {
                            let completed_clone = completed.clone();
                            worker_handle.spawn(async move {
                                simulate_io(id + 10000).await;
                                completed_clone.fetch_add(1, Ordering::Relaxed);
                            });
                        }

                        // Worker 0: occasional CPU-heavy work that blocks the
                        // event loop, creating visible long polls
                        if i == 0 && request_id % 200 == 0 {
                            burn_cpu(2000); // 2ms blocking
                        }

                        request_id += NUM_WORKERS as u64;
                        tokio::task::yield_now().await;
                    }

                    // Drain remaining handoffs
                    handoff_rx.close();
                    while let Some(id) = handoff_rx.recv().await {
                        let completed_clone = completed.clone();
                        worker_handle.spawn(async move {
                            simulate_io(id + 10000).await;
                            completed_clone.fetch_add(1, Ordering::Relaxed);
                        });
                    }

                    // Let in-flight tasks finish
                    tokio::time::sleep(Duration::from_millis(50)).await;
                });
            });
        }
        // Drop extra senders so channels close when workers finish
        drop(handoff_txs);
    });

    let elapsed = start.elapsed();
    let done = completed.load(Ordering::Relaxed);
    drop(guard);

    println!(
        "\n{done} parts in {:.2}s ({:.0} parts/s)",
        elapsed.as_secs_f64(),
        done as f64 / elapsed.as_secs_f64()
    );
    println!("Trace: {TRACE_PATH}");
    println!("View:  python3 dial9-tokio-telemetry/serve.py  →  http://localhost:8000");
    Ok(())
}
