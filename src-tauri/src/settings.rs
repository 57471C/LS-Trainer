//! Persistent app settings (data directory for runs / extraction).

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

use crate::process::get_project_root;

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
    let path = settings_path();
    if !path.is_file() {
        return AppSettings::default();
    }
    match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_default(),
        Err(_) => AppSettings::default(),
    }
}

pub fn save_settings(settings: &AppSettings) -> Result<(), String> {
    let path = settings_path();
    let text = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("Failed to serialize settings: {e}"))?;
    std::fs::write(&path, text).map_err(|e| format!("Failed to write settings: {e}"))?;
    Ok(())
}

/// Returns the absolute base directory for runs.
/// Prefers the configured data directory when it is set and exists (or can be created).
pub fn runs_base_dir() -> PathBuf {
    let settings = load_settings();
    let trimmed = settings.data_directory.trim();
    if !trimmed.is_empty() {
        let p = PathBuf::from(trimmed);
        if p.is_absolute() {
            return p;
        }
    }
    get_project_root()
}

#[tauri::command]
pub async fn get_settings() -> AppSettings {
    load_settings()
}

#[tauri::command]
pub async fn set_data_directory(path: String) -> Result<AppSettings, String> {
    let mut settings = load_settings();
    let trimmed = path.trim().to_string();

    if !trimmed.is_empty() {
        let p = Path::new(&trimmed);
        if !p.is_absolute() {
            return Err("Data directory must be an absolute path".to_string());
        }
        if let Err(e) = std::fs::create_dir_all(p) {
            return Err(format!("Could not create data directory: {e}"));
        }
    }

    settings.data_directory = trimmed;
    save_settings(&settings)?;
    Ok(settings)
}
