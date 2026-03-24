use dial9_tokio_telemetry::telemetry::{
    RotatingWriter, TelemetryEvent, TraceReader, TracedRuntimeGroup, WorkerId,
};
use std::collections::HashSet;
use std::time::Duration;
use tempfile::TempDir;

/// Build a `current_thread` builder with time + IO enabled.
fn current_thread() -> tokio::runtime::Builder {
    let mut b = tokio::runtime::Builder::new_current_thread();
    b.enable_all();
    b
}

/// Read all events from a trace file, including metadata events like
/// `RuntimeDef` that `read_all()` skips.
fn read_all_events(path: &std::path::Path) -> Vec<TelemetryEvent> {
    let reader = TraceReader::new(path.to_str().unwrap()).unwrap();
    reader.events
}

/// Extract `RuntimeDef` events as `(runtime_index, worker_base, worker_count)`
/// sorted by `runtime_index`.
fn runtime_defs(events: &[TelemetryEvent]) -> Vec<(u8, u8, u8)> {
    let mut defs: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            TelemetryEvent::RuntimeDef {
                runtime_index,
                worker_base,
                worker_count,
                ..
            } => Some((*runtime_index, *worker_base, *worker_count)),
            _ => None,
        })
        .collect();
    defs.sort_by_key(|(idx, _, _)| *idx);
    defs
}

/// Collect the set of worker IDs from PollStart events, excluding sentinels.
fn poll_worker_ids(events: &[TelemetryEvent]) -> HashSet<u64> {
    events
        .iter()
        .filter_map(|e| match e {
            TelemetryEvent::PollStart { worker_id, .. } => {
                let id = worker_id.as_u64();
                if id != WorkerId::UNKNOWN.as_u64() && id != WorkerId::BLOCKING.as_u64() {
                    Some(id)
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect()
}

/// Run a spawned task on a runtime. `tokio::spawn` is required because
/// `on_before_task_poll` only fires for spawned tasks, not the root
/// `block_on` future.
async fn do_traced_work() {
    tokio::spawn(async { tokio::task::yield_now().await })
        .await
        .unwrap();
}

#[test]
fn multiple_current_thread_runtimes_produce_unified_trace() {
    let dir = TempDir::new().unwrap();
    let trace_path = dir.path().join("trace.bin");
    let writer = RotatingWriter::single_file(&trace_path).unwrap();

    let mut group = TracedRuntimeGroup::builder()
        .with_task_tracking(true)
        .build(writer)
        .unwrap();

    let rt0 = group.register_named("rt0", current_thread()).unwrap();
    let rt1 = group.register_named("rt1", current_thread()).unwrap();
    let rt2 = group.register_named("rt2", current_thread()).unwrap();

    let guard = group.start();

    // Run work on separate OS threads (can't nest block_on).
    std::thread::scope(|s| {
        s.spawn(|| rt0.block_on(do_traced_work()));
        s.spawn(|| rt1.block_on(do_traced_work()));
        s.spawn(|| rt2.block_on(do_traced_work()));
    });

    drop(rt0);
    drop(rt1);
    drop(rt2);
    drop(guard);

    let events = read_all_events(&trace_path);

    // 3 current_thread runtimes: each gets 1 worker, sequential bases.
    let defs = runtime_defs(&events);
    assert_eq!(
        defs,
        vec![(0, 0, 1), (1, 1, 1), (2, 2, 1)],
        "expected 3 current_thread RuntimeDefs with sequential worker_base; got {defs:?}"
    );

    // PollStart events for workers 0, 1, and 2.
    let workers = poll_worker_ids(&events);
    for w in 0..3u64 {
        assert!(
            workers.contains(&w),
            "expected PollStart events for worker {w}; saw workers {workers:?}"
        );
    }
}

#[test]
fn mixed_multi_thread_and_current_thread_worker_ids_dont_overlap() {
    let dir = TempDir::new().unwrap();
    let trace_path = dir.path().join("trace.bin");
    let writer = RotatingWriter::single_file(&trace_path).unwrap();

    let mut group = TracedRuntimeGroup::builder()
        .with_task_tracking(true)
        .build(writer)
        .unwrap();

    // multi_thread with 2 workers → workers 0, 1
    let mut mt_builder = tokio::runtime::Builder::new_multi_thread();
    mt_builder.worker_threads(2).enable_all();
    let rt_mt = group.register_named("mt", mt_builder).unwrap();

    // current_thread → worker 2
    let rt_ct1 = group.register_named("ct1", current_thread()).unwrap();
    // current_thread → worker 3
    let rt_ct2 = group.register_named("ct2", current_thread()).unwrap();

    let guard = group.start();

    std::thread::scope(|s| {
        s.spawn(|| {
            rt_mt.block_on(async {
                let mut handles = Vec::new();
                for _ in 0..20 {
                    handles.push(tokio::spawn(async { tokio::task::yield_now().await }));
                }
                for h in handles {
                    h.await.unwrap();
                }
            });
        });
        s.spawn(|| rt_ct1.block_on(do_traced_work()));
        s.spawn(|| rt_ct2.block_on(do_traced_work()));
    });

    drop(rt_mt);
    drop(rt_ct1);
    drop(rt_ct2);
    drop(guard);

    let events = read_all_events(&trace_path);

    // mt=0 (2 workers), ct1=2, ct2=3.
    let defs = runtime_defs(&events);
    assert_eq!(
        defs,
        vec![(0, 0, 2), (1, 2, 1), (2, 3, 1)],
        "unexpected RuntimeDef layout: {defs:?}"
    );

    // No worker ID outside the valid range [0..4).
    let workers = poll_worker_ids(&events);
    let valid: HashSet<u64> = (0..4).collect();
    let invalid: HashSet<_> = workers.iter().filter(|w| !valid.contains(w)).collect();
    assert!(
        invalid.is_empty(),
        "found worker IDs outside valid range: {invalid:?}"
    );
}

#[test]
fn queue_samples_tagged_with_runtime_index() {
    let dir = TempDir::new().unwrap();
    let trace_path = dir.path().join("trace.bin");
    let writer = RotatingWriter::single_file(&trace_path).unwrap();

    let mut group = TracedRuntimeGroup::builder().build(writer).unwrap();

    let rt0 = group.register_named("a", current_thread()).unwrap();
    let rt1 = group.register_named("b", current_thread()).unwrap();

    let guard = group.start();

    // Keep runtimes alive long enough for the flush thread to emit queue
    // samples (sampled every ~10ms).
    std::thread::scope(|s| {
        s.spawn(|| {
            rt0.block_on(async { tokio::time::sleep(Duration::from_millis(80)).await });
        });
        s.spawn(|| {
            rt1.block_on(async { tokio::time::sleep(Duration::from_millis(80)).await });
        });
    });

    drop(rt0);
    drop(rt1);
    drop(guard);

    let events = read_all_events(&trace_path);

    let sample_indices: HashSet<u8> = events
        .iter()
        .filter_map(|e| match e {
            TelemetryEvent::QueueSample { runtime_index, .. } => Some(*runtime_index),
            _ => None,
        })
        .collect();

    assert!(
        sample_indices.contains(&0),
        "expected QueueSample with runtime_index=0; got indices {sample_indices:?}"
    );
    assert!(
        sample_indices.contains(&1),
        "expected QueueSample with runtime_index=1; got indices {sample_indices:?}"
    );
}

#[test]
fn register_group_convenience() {
    let dir = TempDir::new().unwrap();
    let trace_path = dir.path().join("trace.bin");
    let writer = RotatingWriter::single_file(&trace_path).unwrap();

    let mut group = TracedRuntimeGroup::builder().build(writer).unwrap();

    let runtimes = group
        .register_group("workers", (0..4).map(|_| current_thread()))
        .unwrap();

    assert_eq!(runtimes.len(), 4);

    let guard = group.start();

    std::thread::scope(|s| {
        for rt in &runtimes {
            s.spawn(|| rt.block_on(do_traced_work()));
        }
    });

    drop(runtimes);
    drop(guard);

    let events = read_all_events(&trace_path);

    // 4 RuntimeDef events with sequential worker IDs 0-3.
    let defs = runtime_defs(&events);
    assert_eq!(
        defs,
        vec![(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)],
        "expected 4 sequential RuntimeDefs; got {defs:?}"
    );

    // PollStart events for all 4 workers.
    let workers = poll_worker_ids(&events);
    for w in 0..4u64 {
        assert!(
            workers.contains(&w),
            "expected PollStart for worker {w}; saw {workers:?}"
        );
    }
}

#[test]
fn cross_runtime_wake_tracking() {
    let dir = TempDir::new().unwrap();
    let trace_path = dir.path().join("trace.bin");
    let writer = RotatingWriter::single_file(&trace_path).unwrap();

    let mut group = TracedRuntimeGroup::builder()
        .with_task_tracking(true)
        .build(writer)
        .unwrap();

    let rt_a = group.register_named("a", current_thread()).unwrap();
    let rt_b = group.register_named("b", current_thread()).unwrap();

    let guard = group.start();
    let handle = guard.handle();

    let notify = std::sync::Arc::new(tokio::sync::Notify::new());

    // Runtime A: spawn a traced task that waits on the Notify.
    let notify_a = notify.clone();
    let handle_a = handle.clone();
    let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_clone = done.clone();

    let t_a = std::thread::spawn(move || {
        rt_a.block_on(async {
            let join = handle_a.spawn(async move {
                notify_a.notified().await;
            });

            // Yield to let the spawned task register its waker.
            tokio::task::yield_now().await;
            done_clone.store(true, std::sync::atomic::Ordering::Release);

            join.await.unwrap();
        });
    });

    // Wait for runtime A's task to be parked on the Notify.
    while !done.load(std::sync::atomic::Ordering::Acquire) {
        std::thread::sleep(Duration::from_millis(1));
    }
    std::thread::sleep(Duration::from_millis(10));

    // Runtime B: notify from a different runtime context.
    let notify_b = notify.clone();
    let t_b = std::thread::spawn(move || {
        rt_b.block_on(async {
            notify_b.notify_one();
            tokio::task::yield_now().await;
        });
    });

    t_a.join().unwrap();
    t_b.join().unwrap();
    drop(guard);

    let events = read_all_events(&trace_path);

    let wake_count = events
        .iter()
        .filter(|e| matches!(e, TelemetryEvent::WakeEvent { .. }))
        .count();

    assert!(
        wake_count > 0,
        "expected at least one WakeEvent for cross-runtime notify; got 0. \
         Total events: {}",
        events.len()
    );
}
