//! Process and path helpers: project root, venv executables, console hiding, and log streaming.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use tauri::ipc::Channel;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

/// Prevent a console window from flashing up for child processes on Windows. No-op elsewhere.
pub(crate) fn hide_console_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(not(windows))]
    let _ = cmd;
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

/// Stream a child's stdout and stderr line-by-line to the log channel and wait for both pipes to close.
pub(crate) fn stream_child_output(child: &mut Child, on_log: &Channel<String>) {
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let log_out = on_log.clone();
    let t_out = std::thread::spawn(move || {
        if let Some(out) = stdout {
            stream_reader_lines(out, &log_out);
        }
    });
    let log_err = on_log.clone();
    let t_err = std::thread::spawn(move || {
        if let Some(err) = stderr {
            stream_reader_lines(err, &log_err);
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

pub(crate) fn stream_reader_lines<R: Read>(mut reader: R, on_log: &Channel<String>) {
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
                    let cleaned = strip_ansi_codes(&raw).trim_end().to_string();
                    if !cleaned.trim().is_empty() && cleaned != last_sent {
                        let _ = on_log.send(cleaned.clone());
                        last_sent = cleaned;
                    }
                    line_buf.clear();
                }
            } else {
                line_buf.push(b);
            }
        }
    }
    if !line_buf.is_empty() {
        let raw = String::from_utf8_lossy(&line_buf);
        let cleaned = strip_ansi_codes(&raw).trim_end().to_string();
        if !cleaned.trim().is_empty() && cleaned != last_sent {
            let _ = on_log.send(cleaned);
        }
    }
}
