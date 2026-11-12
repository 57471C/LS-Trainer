//! Persistent app settings (data directory for runs / extraction).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::process::{get_project_root, to_display_path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    /// Absolute path used as the base for `runs/` (and future extraction).
    /// Empty string means fall back to the project root.
    #[serde(default)]
    pub data_directory: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            data_directory: String::new(),
        }
    }
}

fn settings_path() -> PathBuf {
    get_project_root().join("ls_trainer_settings.json")
}

pub fn load_settings() -> AppSettings {
    read_settings_file(&settings_path())
}

pub(crate) fn read_settings_file(path: &Path) -> AppSettings {
    if !path.is_file() {
        return AppSettings::default();
    }
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => AppSettings::default(),
    }
}

pub fn save_settings(settings: &AppSettings) -> Result<(), String> {
    write_settings_file(&settings_path(), settings)
}

pub(crate) fn write_settings_file(path: &Path, settings: &AppSettings) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create settings directory: {e}"))?;
        }
    }
    let text = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("Failed to serialize settings: {e}"))?;
    std::fs::write(path, text).map_err(|e| format!("Failed to write settings: {e}"))?;
    Ok(())
}

/// Validate and store a data directory on `settings`.
/// An empty path clears the override. A non-empty path must be absolute and is created.
pub(crate) fn apply_data_directory(settings: &mut AppSettings, path: &str) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        settings.data_directory.clear();
        return Ok(());
    }

    let directory = Path::new(trimmed);
    if !directory.is_absolute() {
        return Err("Data directory must be an absolute path".to_string());
    }
    std::fs::create_dir_all(directory).map_err(|err| {
        let mut message = format!(
            "Could not create data directory '{}': {err}.",
            directory.display()
        );
        if err.kind() == std::io::ErrorKind::PermissionDenied {
            if let Some(existing) = nearest_existing_dir(directory) {
                message.push_str(&describe_write_access(&existing));
            }
            message.push_str(" Choose a folder this user can write to.");
        }
        message
    })?;
    ensure_data_directory_writable(directory)?;
    settings.data_directory = to_display_path(directory);
    Ok(())
}

/// Create and remove a probe folder so a read-only mount is rejected at save time.
/// `create_dir_all` succeeds when the directory already exists, even if it is not writable.
pub(crate) fn ensure_data_directory_writable(dir: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!(
            "Data directory '{}' is not a directory.",
            dir.display()
        ));
    }

    let probe = dir.join(format!("ls_trainer_write_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&probe);
    if let Err(err) = std::fs::create_dir(&probe) {
        return Err(writable_error(dir, &err));
    }
    let file_result = std::fs::write(probe.join("write"), b"ok");
    let _ = std::fs::remove_dir_all(&probe);
    file_result.map_err(|err| writable_error(dir, &err))?;
    Ok(())
}

fn writable_error(dir: &Path, err: &std::io::Error) -> String {
    let mut message = format!("Data directory '{}' is not writable: {err}.", dir.display());
    if err.kind() == std::io::ErrorKind::PermissionDenied {
        message.push_str(&describe_write_access(dir));
        message.push_str(
            " Training saves runs inside this folder. Choose a folder this user can write to, or change the folder's owner or mode.",
        );
    }
    message
}

/// Nearest existing directory at `path` or above it.
pub(crate) fn nearest_existing_dir(path: &Path) -> Option<PathBuf> {
    let mut current = path;
    loop {
        match std::fs::metadata(current) {
            Ok(meta) if meta.is_dir() => return Some(current.to_path_buf()),
            Ok(_) => return None,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => match current.parent() {
                Some(parent) if parent != current => current = parent,
                _ => return None,
            },
            Err(_) => return None,
        }
    }
}

/// Owner, mode, and process ids. Empty when the platform cannot read them.
pub(crate) fn describe_write_access(dir: &Path) -> String {
    #[cfg(unix)]
    {
        unix_describe_write_access(dir)
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
        String::new()
    }
}

#[cfg(unix)]
fn unix_describe_write_access(dir: &Path) -> String {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};

    let Ok(meta) = std::fs::metadata(dir) else {
        return String::new();
    };
    let mode = meta.permissions().mode() & 0o777;
    let owner = meta.uid();
    let group = meta.gid();
    let (euid, egid, groups) = linux_process_ids();

    let mut hint = format!(
        " '{}' is mode {mode:04o}, owner uid {owner}, group gid {group}.",
        dir.display()
    );
    if let Some(uid) = euid {
        hint.push_str(&format!(" This process is uid {uid}"));
        if let Some(gid) = egid {
            hint.push_str(&format!(", gid {gid}"));
        }
        if !groups.is_empty() {
            let listed: Vec<String> = groups.iter().map(|gid| gid.to_string()).collect();
            hint.push_str(&format!(", groups {}", listed.join(" ")));
        }
        hint.push('.');
        if uid == 0 {
            hint.push_str(" Root writes are rejected by NFS root_squash. Run LS-Trainer as the user that owns the share.");
        } else if uid != owner && (mode & 0o002) == 0 {
            let in_group = egid == Some(group) || groups.contains(&group);
            if in_group && (mode & 0o020) == 0 {
                hint.push_str(" This user is in the directory group, and the mode does not grant group write.");
            } else if !in_group && (mode & 0o020) != 0 {
                hint.push_str(
                    " The directory is group-writable, and this user is not in that group.",
                );
            } else if !in_group {
                hint.push_str(
                    " This user does not own the directory and is not granted write access.",
                );
            }
        }
    }
    hint
}

#[cfg(unix)]
fn linux_process_ids() -> (Option<u32>, Option<u32>, Vec<u32>) {
    let text = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let mut euid = None;
    let mut egid = None;
    let mut groups = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Uid:") {
            euid = rest
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse().ok());
        } else if let Some(rest) = line.strip_prefix("Gid:") {
            egid = rest
                .split_whitespace()
                .nth(1)
                .and_then(|value| value.parse().ok());
        } else if let Some(rest) = line.strip_prefix("Groups:") {
            groups = rest
                .split_whitespace()
                .filter_map(|value| value.parse().ok())
                .collect();
        }
    }
    (euid, egid, groups)
}

/// Base directory for `runs/`. An absolute configured path wins; otherwise `project_root`.
pub(crate) fn resolve_runs_base(settings: &AppSettings, project_root: &Path) -> PathBuf {
    let trimmed = settings.data_directory.trim();
    if !trimmed.is_empty() {
        let configured = PathBuf::from(trimmed);
        if configured.is_absolute() {
            return configured;
        }
    }
    project_root.to_path_buf()
}

/// Returns the absolute base directory for runs.
/// Prefers the configured data directory when it is an absolute path.
pub fn runs_base_dir() -> PathBuf {
    resolve_runs_base(&load_settings(), &get_project_root())
}

#[tauri::command]
pub async fn get_settings() -> AppSettings {
    load_settings()
}

#[tauri::command]
pub async fn set_data_directory(path: String) -> Result<AppSettings, String> {
    let mut settings = load_settings();
    apply_data_directory(&mut settings, &path)?;
    save_settings(&settings)?;
    Ok(settings)
}
