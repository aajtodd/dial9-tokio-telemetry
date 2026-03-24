use crate::telemetry::buffer::BUFFER;
use crate::telemetry::events::RawEvent;
use crate::telemetry::format::WorkerId;
use crate::telemetry::recorder::event_writer::EventWriter;
use crate::telemetry::recorder::shared_state::{SharedState, resolve_worker_id_with_base};
use crate::telemetry::recorder::{TelemetryGuard, TelemetryRecorder, WorkerHandle};
use crate::telemetry::task_metadata::TaskId;
use crate::telemetry::writer::TraceWriter;
use arc_swap::ArcSwap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::runtime::{Runtime, RuntimeMetrics};

/// Metadata for a registered runtime within a group.
struct RuntimeInfo {
    /// Index of this runtime in the group (0-based).
    runtime_index: u8,
    /// Human-readable name.
    name: String,
    /// First worker ID in the flat range.
    worker_base: u8,
    /// Number of workers.
    worker_count: u8,
    /// "current_thread" or "multi_thread".
    flavor: RuntimeFlavor,
    /// Metrics handle (cloned from `runtime.handle().metrics()`).
    metrics: RuntimeMetrics,
}

#[derive(Clone, Copy)]
enum RuntimeFlavor {
    CurrentThread,
    MultiThread,
}

impl RuntimeFlavor {
    fn as_str(self) -> &'static str {
        match self {
            Self::CurrentThread => "current_thread",
            Self::MultiThread => "multi_thread",
        }
    }
}

/// Builder for [`TracedRuntimeGroup`].
///
/// Construct via [`TracedRuntimeGroup::builder()`].
pub struct TracedRuntimeGroupBuilder {
    task_tracking_enabled: bool,
    trace_path: Option<PathBuf>,
    #[cfg(feature = "cpu-profiling")]
    cpu_profiling_config: Option<crate::telemetry::cpu_profile::CpuProfilingConfig>,
    #[cfg(feature = "cpu-profiling")]
    sched_event_config: Option<crate::telemetry::cpu_profile::SchedEventConfig>,
    #[cfg(feature = "worker-s3")]
    s3_config: Option<crate::background_task::s3::S3Config>,
    #[cfg(feature = "worker-s3")]
    s3_client: Option<aws_sdk_s3::Client>,
    worker_poll_interval: Option<Duration>,
    worker_metrics_sink: Option<metrique_writer::BoxEntrySink>,
}

impl TracedRuntimeGroupBuilder {
    /// Enable task-spawn / task-terminate event recording.
    pub fn with_task_tracking(mut self, enabled: bool) -> Self {
        self.task_tracking_enabled = enabled;
        self
    }

    /// Set the trace output path. Required for background worker support
    /// (S3 upload, offline symbolization).
    pub fn with_trace_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.trace_path = Some(path.into());
        self
    }

    /// Enable CPU profiling (Linux only).
    #[cfg(feature = "cpu-profiling")]
    pub fn with_cpu_profiling(
        mut self,
        config: crate::telemetry::cpu_profile::CpuProfilingConfig,
    ) -> Self {
        self.cpu_profiling_config = Some(config);
        self
    }

    /// Enable scheduler event tracking (Linux only).
    #[cfg(feature = "cpu-profiling")]
    pub fn with_sched_events(
        mut self,
        config: crate::telemetry::cpu_profile::SchedEventConfig,
    ) -> Self {
        self.sched_event_config = Some(config);
        self
    }

    /// Configure S3 upload for sealed trace segments.
    #[cfg(feature = "worker-s3")]
    pub fn with_s3_uploader(mut self, config: crate::background_task::s3::S3Config) -> Self {
        self.s3_config = Some(config);
        self
    }

    /// Provide a pre-built S3 client.
    #[cfg(feature = "worker-s3")]
    pub fn with_s3_client(mut self, client: aws_sdk_s3::Client) -> Self {
        self.s3_client = Some(client);
        self
    }

    /// Set the background worker poll interval.
    pub fn with_worker_poll_interval(mut self, interval: Duration) -> Self {
        self.worker_poll_interval = Some(interval);
        self
    }

    /// Set the metrics sink for flush and worker metrics.
    pub fn with_worker_metrics_sink(mut self, sink: metrique_writer::BoxEntrySink) -> Self {
        self.worker_metrics_sink = Some(sink);
        self
    }

    /// Build the group with the given trace writer.
    ///
    /// After building, call [`TracedRuntimeGroup::register_named`] for each
    /// runtime, then [`TracedRuntimeGroup::start`] to begin recording.
    pub fn build(self, writer: impl TraceWriter + 'static) -> std::io::Result<TracedRuntimeGroup> {
        let start_time_ns = crate::telemetry::events::clock_monotonic_ns();
        let shared = Arc::new(SharedState::new(start_time_ns));
        let recorder = Arc::new(Mutex::new(TelemetryRecorder {
            shared: shared.clone(),
            event_writer: EventWriter::new(Box::new(writer)),
        }));
        Ok(TracedRuntimeGroup {
            shared,
            recorder,
            runtimes: Vec::new(),
            next_worker_id: 0,
            task_tracking_enabled: self.task_tracking_enabled,
            trace_path: self.trace_path,
            #[cfg(feature = "cpu-profiling")]
            cpu_profiling_config: self.cpu_profiling_config,
            #[cfg(feature = "cpu-profiling")]
            sched_event_config: self.sched_event_config,
            #[cfg(feature = "worker-s3")]
            s3_config: self.s3_config,
            #[cfg(feature = "worker-s3")]
            s3_client: self.s3_client,
            worker_poll_interval: self.worker_poll_interval,
            worker_metrics_sink: self.worker_metrics_sink,
        })
    }
}

/// Manages telemetry across multiple Tokio runtimes sharing a single trace
/// stream.
///
/// Each runtime is assigned a contiguous range of flat worker IDs so that
/// events from different runtimes can be distinguished in the trace.
///
/// # Usage
///
/// ```rust,no_run
/// # use dial9_tokio_telemetry::telemetry::{TracedRuntimeGroup, NullWriter};
/// let mut group = TracedRuntimeGroup::builder()
///     .with_task_tracking(true)
///     .build(NullWriter)?;
///
/// let rt1 = group.register_named(
///     "api",
///     tokio::runtime::Builder::new_multi_thread(),
/// )?;
/// let rt2 = group.register_named(
///     "background",
///     tokio::runtime::Builder::new_current_thread(),
/// )?;
///
/// let guard = group.start();
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct TracedRuntimeGroup {
    shared: Arc<SharedState>,
    recorder: Arc<Mutex<TelemetryRecorder>>,
    runtimes: Vec<RuntimeInfo>,
    next_worker_id: u8,
    task_tracking_enabled: bool,
    trace_path: Option<PathBuf>,
    #[cfg(feature = "cpu-profiling")]
    cpu_profiling_config: Option<crate::telemetry::cpu_profile::CpuProfilingConfig>,
    #[cfg(feature = "cpu-profiling")]
    sched_event_config: Option<crate::telemetry::cpu_profile::SchedEventConfig>,
    #[cfg(feature = "worker-s3")]
    s3_config: Option<crate::background_task::s3::S3Config>,
    #[cfg(feature = "worker-s3")]
    s3_client: Option<aws_sdk_s3::Client>,
    worker_poll_interval: Option<Duration>,
    worker_metrics_sink: Option<metrique_writer::BoxEntrySink>,
}

impl TracedRuntimeGroup {
    /// Create a new builder.
    pub fn builder() -> TracedRuntimeGroupBuilder {
        TracedRuntimeGroupBuilder {
            task_tracking_enabled: false,
            trace_path: None,
            #[cfg(feature = "cpu-profiling")]
            cpu_profiling_config: None,
            #[cfg(feature = "cpu-profiling")]
            sched_event_config: None,
            #[cfg(feature = "worker-s3")]
            s3_config: None,
            #[cfg(feature = "worker-s3")]
            s3_client: None,
            worker_poll_interval: None,
            worker_metrics_sink: None,
        }
    }

    /// Register a named runtime. Installs telemetry callbacks on the builder,
    /// builds the runtime, and returns it. The group retains metadata for
    /// the flush thread and `RuntimeDef` emission.
    ///
    /// For `current_thread` builders, callbacks use a captured constant worker
    /// ID (no TLS lookup, no metrics scan). The flavor is detected automatically
    /// after `build()` from the worker count.
    pub fn register_named(
        &mut self,
        name: &str,
        mut builder: tokio::runtime::Builder,
    ) -> std::io::Result<Runtime> {
        let runtime_index = self.runtimes.len() as u8;
        let worker_base = self.next_worker_id;

        // We don't know the flavor yet, so install both paths behind a flag.
        let rt_metrics = self.install_callbacks(&mut builder, worker_base, false);

        let runtime = builder.build()?;
        let metrics = runtime.handle().metrics();
        let worker_count = metrics.num_workers();

        let flavor = if worker_count == 1 {
            RuntimeFlavor::CurrentThread
        } else {
            RuntimeFlavor::MultiThread
        };

        rt_metrics.store(Arc::new(Some(metrics.clone())));
        self.next_worker_id = worker_base + worker_count as u8;

        self.runtimes.push(RuntimeInfo {
            runtime_index,
            name: name.to_string(),
            worker_base,
            worker_count: worker_count as u8,
            flavor,
            metrics,
        });

        Ok(runtime)
    }

    /// Register a `current_thread` runtime with optimized callbacks.
    ///
    /// The worker ID is captured as a constant in the callbacks — no TLS
    /// lookup, no `ArcSwap` load, no metrics scan on the hot path. This is
    /// strictly cheaper than [`register_named`](Self::register_named) for
    /// `current_thread` runtimes.
    ///
    /// # Panics
    ///
    /// Panics if the built runtime has more than one worker (i.e., the builder
    /// was not created with `Builder::new_current_thread()`).
    pub fn register_current_thread(
        &mut self,
        name: &str,
        mut builder: tokio::runtime::Builder,
    ) -> std::io::Result<Runtime> {
        let runtime_index = self.runtimes.len() as u8;
        let worker_base = self.next_worker_id;

        let rt_metrics = self.install_callbacks(&mut builder, worker_base, true);

        let runtime = builder.build()?;
        let metrics = runtime.handle().metrics();
        let worker_count = metrics.num_workers();
        assert_eq!(
            worker_count, 1,
            "register_current_thread requires a current_thread runtime, got {worker_count} workers"
        );

        rt_metrics.store(Arc::new(Some(metrics.clone())));
        self.next_worker_id = worker_base + 1;

        self.runtimes.push(RuntimeInfo {
            runtime_index,
            name: name.to_string(),
            worker_base,
            worker_count: 1,
            flavor: RuntimeFlavor::CurrentThread,
            metrics,
        });

        Ok(runtime)
    }

    /// Register a batch of `current_thread` runtimes as a named group.
    ///
    /// Each runtime is named `"{name}-{i}"` where `i` is its zero-based
    /// index within the batch. Uses the optimized constant worker ID path
    /// (see [`register_current_thread`](Self::register_current_thread)).
    pub fn register_group(
        &mut self,
        name: &str,
        builders: impl IntoIterator<Item = tokio::runtime::Builder>,
    ) -> std::io::Result<Vec<Runtime>> {
        builders
            .into_iter()
            .enumerate()
            .map(|(i, builder)| self.register_current_thread(&format!("{name}-{i}"), builder))
            .collect()
    }

    /// Consume the group and start the flush thread. Returns a
    /// [`TelemetryGuard`] that controls recording lifetime.
    pub fn start(self) -> TelemetryGuard {
        self.shared.enabled.store(true, Ordering::Relaxed);

        // Set up CPU profiling if configured.
        #[cfg(feature = "cpu-profiling")]
        {
            let mut rec = self.recorder.lock().unwrap();
            if let Some(ref config) = self.cpu_profiling_config {
                if let Ok(sampler) =
                    crate::telemetry::cpu_profile::CpuProfiler::start(config.clone())
                {
                    rec.event_writer.cpu_profiler = Some(sampler);
                }
            }
            if let Some(config) = self.sched_event_config {
                if let Ok(sched) = crate::telemetry::cpu_profile::SchedProfiler::new(config) {
                    *rec.shared.sched_profiler.lock().unwrap() = Some(sched);
                }
            }
        }

        // Emit RuntimeDef events for each registered runtime.
        // Write directly through the EventWriter (not the buffer/collector path)
        // because the calling thread's buffer won't be flushed before the flush
        // thread starts.
        {
            let ts = crate::telemetry::events::clock_monotonic_ns();
            let mut rec = self.recorder.lock().unwrap();
            for rt in &self.runtimes {
                let event = RawEvent::RuntimeDef {
                    timestamp_nanos: ts,
                    runtime_index: rt.runtime_index,
                    name: rt.name.clone(),
                    worker_base: rt.worker_base,
                    worker_count: rt.worker_count,
                    flavor: rt.flavor.as_str().to_string(),
                };
                if let Err(e) = rec.event_writer.write_raw_event(event) {
                    tracing::warn!("failed to write RuntimeDef: {e}");
                }
            }
        }

        let stop = Arc::new(AtomicBool::new(false));
        let rec = self.recorder.clone();
        let shared = self.shared.clone();
        let runtimes = self.runtimes;
        let stop_clone = stop.clone();

        let thread = std::thread::Builder::new()
            .name("telemetry-flush".into())
            .spawn(move || {
                // Lower this thread's scheduling priority so it doesn't
                // compete with worker threads for CPU time.
                #[cfg(target_os = "linux")]
                unsafe {
                    let _ = libc::nice(10);
                }

                let sample_interval = Duration::from_millis(10);
                let mut last_sample = Instant::now();

                while !stop_clone.load(Ordering::Acquire) {
                    std::thread::sleep(Duration::from_millis(5));

                    let now = Instant::now();
                    if now.duration_since(last_sample) >= sample_interval {
                        last_sample = now;
                        for rt in &runtimes {
                            shared.record_queue_sample(
                                rt.runtime_index,
                                rt.metrics.global_queue_depth(),
                            );
                        }
                    }

                    // Flush the flush thread's own buffer (queue-sample events)
                    // so they reach the collector before we drain it.
                    BUFFER.with(|buf| {
                        let mut buf = buf.borrow_mut();
                        let events = buf.flush();
                        if !events.is_empty() {
                            shared.collector.accept_flush(events);
                        }
                    });

                    rec.lock().unwrap().flush();

                    // After flush, check if the writer rotated to a new file.
                    // If so, re-emit RuntimeDef events so each segment is
                    // self-describing.
                    {
                        let mut recorder = rec.lock().unwrap();
                        if recorder.event_writer.writer.take_rotated() {
                            let ts = crate::telemetry::events::clock_monotonic_ns();
                            for rt in &runtimes {
                                let event = RawEvent::RuntimeDef {
                                    timestamp_nanos: ts,
                                    runtime_index: rt.runtime_index,
                                    name: rt.name.clone(),
                                    worker_base: rt.worker_base,
                                    worker_count: rt.worker_count,
                                    flavor: rt.flavor.as_str().to_string(),
                                };
                                if let Err(e) = recorder.event_writer.write_raw_event(event) {
                                    tracing::warn!(
                                        "failed to re-emit RuntimeDef after rotation: {e}"
                                    );
                                }
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn telemetry-flush thread");

        // Spawn background worker for S3 upload / symbolization if configured.
        let worker = self.trace_path.and_then(|trace_path| {
            #[allow(unused_mut)]
            let mut needs_worker = false;
            #[allow(unused_mut)]
            let mut symbolize = false;

            #[cfg(feature = "cpu-profiling")]
            if self.cpu_profiling_config.is_some() {
                needs_worker = true;
                symbolize = true;
            }

            #[cfg(feature = "worker-s3")]
            let s3 = self.s3_config;
            #[cfg(feature = "worker-s3")]
            if s3.is_some() {
                needs_worker = true;
            }

            if !needs_worker {
                return None;
            }

            let poll_interval = self
                .worker_poll_interval
                .unwrap_or(crate::background_task::DEFAULT_POLL_INTERVAL);
            let metrics_sink = self
                .worker_metrics_sink
                .unwrap_or_else(metrique_writer::sink::DevNullSink::boxed);

            let config = crate::background_task::BackgroundTaskConfig::builder()
                .trace_path(trace_path)
                .poll_interval(poll_interval)
                .symbolize(symbolize)
                .metrics_sink(metrics_sink);

            #[cfg(feature = "worker-s3")]
            let config = config.maybe_s3(s3).maybe_client(self.s3_client);

            let config = config.build();
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
            let wt = std::thread::Builder::new()
                .name("dial9-worker".into())
                .spawn(move || {
                    crate::background_task::run_background_task(config, shutdown_rx);
                })
                .expect("failed to spawn dial9-worker thread");
            Some(WorkerHandle {
                shutdown: Some(shutdown_tx),
                thread: Some(wt),
            })
        });

        TelemetryGuard::new_with_worker(self.shared, self.recorder, stop, Some(thread), worker)
    }

    /// Install the four core callbacks (park, unpark, poll_start, poll_end)
    /// and optionally task-tracking callbacks on the builder.
    ///
    /// Returns a per-runtime metrics handle that must be populated after
    /// `builder.build()`.
    ///
    /// When `current_thread_hint` is true, callbacks capture the worker ID as
    /// a constant (`worker_base`) and skip TLS resolution and metrics loads.
    /// This is strictly cheaper than the general path.
    fn install_callbacks(
        &self,
        builder: &mut tokio::runtime::Builder,
        worker_base: u8,
        current_thread_hint: bool,
    ) -> Arc<ArcSwap<Option<RuntimeMetrics>>> {
        let shared = self.shared.clone();
        let base = worker_base as usize;
        let rt_metrics: Arc<ArcSwap<Option<RuntimeMetrics>>> =
            Arc::new(ArcSwap::from_pointee(None));

        if current_thread_hint {
            // current_thread optimization: worker ID is a captured constant.
            // No TLS lookup, no ArcSwap load, no metrics scan.
            let wid = WorkerId::from(base);

            let s = shared.clone();
            builder.on_thread_park(move || {
                let cpu_time_nanos = crate::telemetry::events::thread_cpu_time_nanos();
                if let Ok(ss) = crate::telemetry::events::SchedStat::read_current() {
                    super::shared_state::PARKED_SCHED_WAIT.with(|c| c.set(ss.wait_time_ns));
                }
                s.record_event(RawEvent::WorkerPark {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: wid,
                    worker_local_queue_depth: 0,
                    cpu_time_nanos,
                });
            });

            let s = shared.clone();
            builder.on_thread_unpark(move || {
                let cpu_time_nanos = crate::telemetry::events::thread_cpu_time_nanos();
                let sched_wait_delta_nanos =
                    if let Ok(ss) = crate::telemetry::events::SchedStat::read_current() {
                        let prev = super::shared_state::PARKED_SCHED_WAIT.with(|c| c.get());
                        ss.wait_time_ns.saturating_sub(prev)
                    } else {
                        0
                    };
                s.record_event(RawEvent::WorkerUnpark {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: wid,
                    worker_local_queue_depth: 0,
                    cpu_time_nanos,
                    sched_wait_delta_nanos,
                });
            });

            let s = shared.clone();
            builder.on_before_task_poll(move |meta| {
                let task_id = TaskId::from(meta.id());
                let location = meta.spawned_at();
                s.record_event(RawEvent::PollStart {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: wid,
                    worker_local_queue_depth: 0,
                    task_id,
                    location,
                });
            });

            let s = shared.clone();
            builder.on_after_task_poll(move |_meta| {
                s.record_event(RawEvent::PollEnd {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: wid,
                });
            });
        } else {
            // multi_thread path: resolve worker ID via TLS + metrics scan.
            let s = shared.clone();
            let m = rt_metrics.clone();
            builder.on_thread_park(move || {
                let worker_id = resolve_worker_id_with_base(&m, Some(&s), base);
                let metrics_guard = m.load();
                let local_q = match (worker_id, &**metrics_guard) {
                    (Some(wid), Some(metrics)) => metrics.worker_local_queue_depth(wid - base),
                    _ => 0,
                };
                let cpu_time_nanos = crate::telemetry::events::thread_cpu_time_nanos();
                if let Ok(ss) = crate::telemetry::events::SchedStat::read_current() {
                    super::shared_state::PARKED_SCHED_WAIT.with(|c| c.set(ss.wait_time_ns));
                }
                s.record_event(RawEvent::WorkerPark {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: worker_id.map(WorkerId::from).unwrap_or(WorkerId::UNKNOWN),
                    worker_local_queue_depth: local_q,
                    cpu_time_nanos,
                });
            });

            let s = shared.clone();
            let m = rt_metrics.clone();
            builder.on_thread_unpark(move || {
                let worker_id = resolve_worker_id_with_base(&m, Some(&s), base);
                let metrics_guard = m.load();
                let local_q = match (worker_id, &**metrics_guard) {
                    (Some(wid), Some(metrics)) => metrics.worker_local_queue_depth(wid - base),
                    _ => 0,
                };
                let cpu_time_nanos = crate::telemetry::events::thread_cpu_time_nanos();
                let sched_wait_delta_nanos =
                    if let Ok(ss) = crate::telemetry::events::SchedStat::read_current() {
                        let prev = super::shared_state::PARKED_SCHED_WAIT.with(|c| c.get());
                        ss.wait_time_ns.saturating_sub(prev)
                    } else {
                        0
                    };
                s.record_event(RawEvent::WorkerUnpark {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: worker_id.map(WorkerId::from).unwrap_or(WorkerId::UNKNOWN),
                    worker_local_queue_depth: local_q,
                    cpu_time_nanos,
                    sched_wait_delta_nanos,
                });
            });

            let s = shared.clone();
            let m = rt_metrics.clone();
            builder.on_before_task_poll(move |meta| {
                let task_id = TaskId::from(meta.id());
                let location = meta.spawned_at();
                let worker_id = resolve_worker_id_with_base(&m, Some(&s), base);
                let metrics_guard = m.load();
                let local_q = match (worker_id, &**metrics_guard) {
                    (Some(wid), Some(metrics)) => metrics.worker_local_queue_depth(wid - base),
                    _ => 0,
                };
                s.record_event(RawEvent::PollStart {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: worker_id.map(WorkerId::from).unwrap_or(WorkerId::UNKNOWN),
                    worker_local_queue_depth: local_q,
                    task_id,
                    location,
                });
            });

            let s = shared.clone();
            let m = rt_metrics.clone();
            builder.on_after_task_poll(move |_meta| {
                let worker_id = resolve_worker_id_with_base(&m, Some(&s), base);
                s.record_event(RawEvent::PollEnd {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    worker_id: worker_id.map(WorkerId::from).unwrap_or(WorkerId::UNKNOWN),
                });
            });
        }

        // -- task tracking --
        if self.task_tracking_enabled {
            let s = shared.clone();
            builder.on_task_spawn(move |meta| {
                let task_id = TaskId::from(meta.id());
                let location = meta.spawned_at();
                s.record_event(RawEvent::TaskSpawn {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    task_id,
                    location,
                });
            });
            let s = shared;
            builder.on_task_terminate(move |meta| {
                let task_id = TaskId::from(meta.id());
                s.record_event(RawEvent::TaskTerminate {
                    timestamp_nanos: crate::telemetry::events::clock_monotonic_ns(),
                    task_id,
                });
            });
        }

        // -- cpu-profiling thread tracking --
        #[cfg(feature = "cpu-profiling")]
        {
            let s_start = self.shared.clone();
            let s_stop = self.shared.clone();
            builder
                .on_thread_start(move || {
                    {
                        let tid = crate::telemetry::events::current_tid();
                        s_start
                            .thread_roles
                            .lock()
                            .unwrap()
                            .insert(tid, crate::telemetry::events::ThreadRole::Blocking);
                    }
                    if let Ok(mut prof) = s_start.sched_profiler.lock()
                        && let Some(ref mut p) = *prof
                    {
                        let _ = p.track_current_thread();
                    }
                })
                .on_thread_stop(move || {
                    {
                        let tid = crate::telemetry::events::current_tid();
                        s_stop.thread_roles.lock().unwrap().remove(&tid);
                    }
                    if let Ok(mut prof) = s_stop.sched_profiler.lock()
                        && let Some(ref mut p) = *prof
                    {
                        p.stop_tracking_current_thread();
                    }
                });
        }

        rt_metrics
    }
}
