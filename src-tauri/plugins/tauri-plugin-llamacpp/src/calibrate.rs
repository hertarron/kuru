//! One-model probe load that reports what llama.cpp actually allocates.
//!
//! The context planner sizes the KV cache and the weights from GGUF metadata
//! exactly, but the compute buffer is decided by llama.cpp's graph builder and
//! is not derivable from the header: three models measured at the same context
//! and micro-batch came out at 441, 92 and 156 MiB with no ordering by
//! embedding size, layer count or expert count. So the planner reserves the
//! largest figure ever seen until a model has been measured, and this is the
//! measurement.
//!
//! The numbers are printed only at verbosity 5, and only into `--log-file`:
//! the router forwards its child's stdout, but the buffer lines are gone from
//! that stream by the time the parent sees it. Verbosity 5 on the app's own
//! router would flood the log with per-request lines, which is why this runs
//! as a separate short-lived process instead.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use tauri::{Manager, Runtime, State};
use tokio::process::Command;
use tokio::time::{sleep, Instant};

use crate::error::{ServerError, ServerResult};
use crate::router::force_kill_router_tree_by_pid;
use crate::state::LlamacppState;
use jan_utils::{find_cuda_paths, find_rocm_paths, setup_library_path, setup_windows_process_flags};

/// Lines that end the probe. The load has either finished or failed by then;
/// nothing after it is measured.
fn is_terminal_line(lower: &str) -> bool {
    lower.contains("model loaded")
        || lower.contains("cudamalloc failed")
        || lower.contains("failed to allocate")
        || lower.contains("error loading model")
        || lower.contains("failed to load model")
        || lower.contains("unable to load model")
}

/// Kept from the log: the per-device allocations, and whatever failed.
fn is_interesting_line(lower: &str) -> bool {
    lower.contains("buffer size")
        || lower.contains("cudamalloc")
        || lower.contains("error loading model")
        || lower.contains("failed to load model")
        || lower.contains("failed to allocate")
}

fn read_log(path: &Path) -> String {
    std::fs::read(path)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .unwrap_or_default()
}

/// Cancels a running calibration probe, if any. Sets the flag the
/// `calibrate_model` wait loop polls and kills the probe tree, so a stuck
/// load releases VRAM/RAM immediately instead of hanging until the timeout.
/// Returns true when a probe was actually running.
#[tauri::command]
pub async fn cancel_calibrate<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
) -> Result<bool, String> {
    let state: State<Arc<LlamacppState>> = app_handle.state();
    state
        .calibrate_cancel
        .store(true, Ordering::SeqCst);
    let pid = state.calibrate_pid.swap(0, Ordering::SeqCst);
    if pid != 0 {
        force_kill_router_tree_by_pid(pid);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Load `preset_path`'s single model once and return the allocation lines it
/// printed. The child is always killed before returning, including on error
/// and on cancellation (see `cancel_calibrate`).
///
/// `preset_path` must name exactly one model with `load-on-startup = true`:
/// the router loads lazily otherwise, and a probe that sends no request would
/// measure nothing.
#[tauri::command]
pub async fn calibrate_model<R: Runtime>(
    app_handle: tauri::AppHandle<R>,
    backend_exe: PathBuf,
    preset_path: PathBuf,
    log_path: PathBuf,
    port: u16,
    envs: HashMap<String, String>,
    timeout_secs: Option<u64>,
) -> ServerResult<Vec<String>> {
    let state: State<Arc<LlamacppState>> = app_handle.state();
    state.calibrate_cancel.store(false, Ordering::SeqCst);
    let args: Vec<String> = vec![
        "--models-preset".to_string(),
        preset_path.to_string_lossy().to_string(),
        "--models-max".to_string(),
        "1".to_string(),
        "--host".to_string(),
        "127.0.0.1".to_string(),
        "--port".to_string(),
        port.to_string(),
        "--no-ui".to_string(),
        "--log-file".to_string(),
        log_path.to_string_lossy().to_string(),
        // The allocation lines exist only at this verbosity.
        "-lv".to_string(),
        "5".to_string(),
    ];
    log::info!("Calibration probe argv: {:?}", args);

    // Left behind by an earlier probe, and llama.cpp appends nothing until it
    // opens the file, so a stale copy would be parsed as this run's result.
    let _ = std::fs::remove_file(&log_path);

    let mut command = Command::new(&backend_exe);
    command.args(&args);
    command.envs(&envs);
    command.stdout(Stdio::null());
    command.stderr(Stdio::null());
    command.kill_on_drop(true);
    setup_windows_process_flags(&mut command);
    let cuda = find_cuda_paths();
    let rocm = find_rocm_paths();
    setup_library_path(backend_exe.parent(), &cuda.merged(rocm), &mut command);

    let mut child = command.spawn().map_err(ServerError::Io)?;
    let pid = child.id();
    state
        .calibrate_pid
        .store(pid.unwrap_or(0), Ordering::SeqCst);

    let deadline = Instant::now() + Duration::from_secs(timeout_secs.unwrap_or(300));
    let mut cancelled = false;
    loop {
        sleep(Duration::from_millis(250)).await;
        if state.calibrate_cancel.load(Ordering::SeqCst) {
            cancelled = true;
            log::info!("Calibration probe cancelled by user");
            break;
        }
        if read_log(&log_path)
            .lines()
            .any(|line| is_terminal_line(&line.to_lowercase()))
        {
            break;
        }
        // A process that exited without a terminal line failed early; its log
        // still carries the reason.
        if matches!(child.try_wait(), Ok(Some(_))) {
            break;
        }
        if Instant::now() >= deadline {
            log::warn!("Calibration probe timed out after {:?}", deadline);
            break;
        }
    }

    // The router spawns a per-model child of its own, which outlives its
    // parent unless the whole tree goes.
    state.calibrate_pid.store(0, Ordering::SeqCst);
    if let Some(pid) = pid {
        force_kill_router_tree_by_pid(pid);
    }
    let _ = child.kill().await;

    if cancelled {
        return Err(ServerError::InvalidArgument(
            "Calibration cancelled by user.".into(),
        ));
    }

    // The child holds the log open while it runs, so the last lines can land
    // after the kill.
    sleep(Duration::from_millis(250)).await;

    Ok(read_log(&log_path)
        .lines()
        .filter(|line| is_interesting_line(&line.to_lowercase()))
        .map(|line| line.trim().to_string())
        .collect())
}
