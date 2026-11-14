//! Process and path helpers: project root, venv executables, console hiding, event sinks, and log/metrics streaming.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command};
use std::sync::{Arc, Mutex};
use tauri::ipc::Channel;

pub use crate::models::TrainMetrics;

pub const METRICS_PREFIX: &str = "@@LS_METRICS@@";

/// Cross-platform event sink for decoupled training logs, telemetry metrics, and process exit events.
pub trait TrainingEventSink: Send + Sync + 'static {
    fn on_log(&self, line: String);
    fn on_metrics(&self, metrics: TrainMetrics);
    fn on_exit(&self, code: Option<i32>);
    fn preserve_ansi(&self) -> bool {
        false
    }
}

/// Event sink adapter wrapping Tauri's `Channel<String>` so existing GUI commands work identically.
#[derive(Clone)]
pub struct TauriChannelSink {
    channel: Channel<String>,
}

impl TauriChannelSink {
    pub fn new(channel: Channel<String>) -> Self {
        Self { channel }
    }

    pub fn channel(&self) -> &Channel<String> {
        &self.channel
    }
}

impl TrainingEventSink for TauriChannelSink {
    fn on_log(&self, line: String) {
        let _ = self.channel.send(line);
    }

    fn on_metrics(&self, _metrics: TrainMetrics) {}

    fn on_exit(&self, _code: Option<i32>) {}
}

/// Control commands supported by the Python training starters over piped stdin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessControlCommand {
    Pause,
    Resume,
    Checkpoint,
    Stop,
}

impl ProcessControlCommand {
    pub fn as_wire_line(&self) -> &'static str {
        match self {
            Self::Pause => "PAUSE\n",
            Self::Resume => "RESUME\n",
            Self::Checkpoint => "CHECKPOINT\n",
            Self::Stop => "STOP\n",
        }
    }
}

/// Thread-safe handle on a running child process's piped `stdin` for sending control commands
/// (`PAUSE\n`, `RESUME\n`, `CHECKPOINT\n`, `STOP\n`) on both Windows and Linux.
#[derive(Clone, Default)]
pub struct ProcessControlHandle {
    stdin: Arc<Mutex<Option<ChildStdin>>>,
}

impl ProcessControlHandle {
    pub fn new(stdin: Option<ChildStdin>) -> Self {
        Self {
            stdin: Arc::new(Mutex::new(stdin)),
        }
    }

    pub fn from_child(child: &mut Child) -> Self {
        Self::new(child.stdin.take())
    }

    pub fn attach(&self, stdin: ChildStdin) {
        if let Ok(mut guard) = self.stdin.lock() {
            *guard = Some(stdin);
        }
    }

    pub fn clear(&self) {
        if let Ok(mut guard) = self.stdin.lock() {
            *guard = None;
        }
    }

    pub fn write_raw(&self, command_line: &str) -> Result<(), String> {
        let mut guard = self
            .stdin
            .lock()
            .map_err(|_| "Failed to lock child stdin handle".to_string())?;
        let Some(stdin) = guard.as_mut() else {
            return Err("No active child stdin pipe attached".to_string());
        };
        let payload = if command_line.ends_with('\n') {
            command_line.to_string()
        } else {
            format!("{command_line}\n")
        };
        stdin
            .write_all(payload.as_bytes())
            .and_then(|_| stdin.flush())
            .map_err(|e| format!("Failed to write control command to child stdin: {e}"))
    }

    pub fn send_command(&self, cmd: ProcessControlCommand) -> Result<(), String> {
        self.write_raw(cmd.as_wire_line())
    }

    pub fn pause(&self) -> Result<(), String> {
        self.send_command(ProcessControlCommand::Pause)
    }

    pub fn resume(&self) -> Result<(), String> {
        self.send_command(ProcessControlCommand::Resume)
    }

    pub fn checkpoint(&self) -> Result<(), String> {
        self.send_command(ProcessControlCommand::Checkpoint)
    }

    pub fn stop(&self) -> Result<(), String> {
        self.send_command(ProcessControlCommand::Stop)
    }
}

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;
#[cfg(windows)]
const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;

/// Prevent a console window from flashing up for child processes on Windows. No-op elsewhere.
pub(crate) fn hide_console_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }
    #[cfg(not(windows))]
    let _ = cmd;
}

#[cfg(windows)]
unsafe extern "system" {
    fn GenerateConsoleCtrlEvent(dwCtrlEvent: u32, dwProcessGroupId: u32) -> i32;
}

/// Send a graceful break / interrupt signal to the target child process group.
pub(crate) fn send_break_signal(pid: u32) -> bool {
    #[cfg(windows)]
    {
        // 1 = CTRL_BREAK_EVENT
        unsafe { GenerateConsoleCtrlEvent(1, pid) != 0 }
    }
    #[cfg(not(windows))]
    {
        let res = std::process::Command::new("kill")
            .args(["-2", &pid.to_string()])
            .status();
        res.map(|s| s.success()).unwrap_or(false)
    }
}

/// Path to the project venv interpreter (`.venv\Scripts\python.exe` on Windows, `.venv/bin/python` elsewhere).
pub(crate) fn venv_python_path(project_root: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    let p = project_root.join(".venv").join("Scripts").join("python.exe");
    #[cfg(not(target_os = "windows"))]
    let p = project_root.join(".venv").join("bin").join("python");
    p
}

/// Path to the project venv Ultralytics CLI (`yolo.exe` on Windows, `yolo` elsewhere).
pub(crate) fn venv_yolo_path(project_root: &Path) -> PathBuf {
    #[cfg(target_os = "windows")]
    let p = project_root.join(".venv").join("Scripts").join("yolo.exe");
    #[cfg(not(target_os = "windows"))]
    let p = project_root.join(".venv").join("bin").join("yolo");
    p
}

/// Name of the environment setup script for the current platform.
pub(crate) fn setup_script_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "setup-venv.bat"
    } else {
        "setup-linux.sh"
    }
}

/// Stream a child's stdout and stderr line-by-line to the event sink and wait for both pipes to close.
pub(crate) fn stream_child_output(child: &mut Child, sink: &Arc<dyn TrainingEventSink>) {
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let sink_out = Arc::clone(sink);
    let t_out = std::thread::spawn(move || {
        if let Some(out) = stdout {
            stream_reader_lines(out, sink_out.as_ref());
        }
    });
    let sink_err = Arc::clone(sink);
    let t_err = std::thread::spawn(move || {
        if let Some(err) = stderr {
            stream_reader_lines(err, sink_err.as_ref());
        }
    });
    let _ = t_out.join();
    let _ = t_err.join();
}

pub(crate) fn get_project_root() -> PathBuf {
    if let Ok(cwd) = std::env::current_dir() {
        if cwd.join(".venv").exists()
            || cwd.join("src-tauri").exists()
            || cwd.join("package.json").exists()
        {
            return cwd;
        }
        if let Some(parent) = cwd.parent() {
            if parent.join(".venv").exists()
                || parent.join("src-tauri").exists()
                || parent.join("package.json").exists()
            {
                return parent.to_path_buf();
            }
        }
    }

    if let Ok(exe_path) = std::env::current_exe() {
        let mut cur = exe_path.parent();
        while let Some(dir) = cur {
            if dir.join(".venv").exists()
                || dir.join("src-tauri").exists()
                || dir.join("package.json").exists()
            {
                return dir.to_path_buf();
            }
            cur = dir.parent();
        }
    }

    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn to_display_path(path: &Path) -> String {
    #[cfg(target_os = "windows")]
    {
        path.to_string_lossy().replace('/', "\\")
    }
    #[cfg(not(target_os = "windows"))]
    {
        path.to_string_lossy().replace('\\', "/")
    }
}

pub(crate) fn get_venv_python(project_root: &Path) -> Option<PathBuf> {
    let p = venv_python_path(project_root);

    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

pub(crate) fn strip_ansi_codes(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&nc) = chars.peek() {
                    chars.next();
                    if ('@'..='~').contains(&nc) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn process_raw_output_line(raw: &str, last_sent: &mut String, sink: &dyn TrainingEventSink) {
    let trimmed = raw.trim();
    if let Some(json_str) = trimmed.strip_prefix(METRICS_PREFIX) {
        if let Ok(metrics) = serde_json::from_str::<TrainMetrics>(json_str.trim()) {
            sink.on_metrics(metrics);
        }
        return;
    }

    let cleaned = strip_ansi_codes(raw).trim_end().to_string();
    if !cleaned.trim().is_empty() && cleaned != *last_sent {
        if sink.preserve_ansi() {
            sink.on_log(raw.trim_end().to_string());
        } else {
            sink.on_log(cleaned.clone());
        }
        *last_sent = cleaned;
    }
}

pub(crate) fn stream_reader_lines<R: Read>(mut reader: R, sink: &dyn TrainingEventSink) {
    let mut buf = [0u8; 1024];
    let mut line_buf = Vec::with_capacity(256);
    let mut last_sent = String::new();

    while let Ok(n) = reader.read(&mut buf) {
        if n == 0 {
            break;
        }
        for &b in &buf[..n] {
            if b == b'\n' || b == b'\r' {
                if !line_buf.is_empty() {
                    let raw = String::from_utf8_lossy(&line_buf);
                    process_raw_output_line(&raw, &mut last_sent, sink);
                    line_buf.clear();
                }
            } else {
                line_buf.push(b);
            }
        }
    }
    if !line_buf.is_empty() {
        let raw = String::from_utf8_lossy(&line_buf);
        process_raw_output_line(&raw, &mut last_sent, sink);
    }
}
