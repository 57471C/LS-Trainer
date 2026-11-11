//! Persistent app settings (data directory for runs / extraction).

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

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
    std::fs::create_dir_all(directory)
        .map_err(|e| format!("Could not create data directory: {e}"))?;
    settings.data_directory = to_display_path(directory);
    Ok(())
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
