//! Dataset preparation and inspection.

use std::path::PathBuf;

pub(crate) mod coco;
pub(crate) mod labels;
pub(crate) mod store;
pub(crate) mod yolo;

fn current_home_dir() -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .filter(|v| !v.is_empty())
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME").ok().filter(|v| !v.is_empty())
    }
}

/// Expand a leading `~`, `~/`, `~\`, `$HOME`, or `%USERPROFILE%` in a dataset path.
/// Absolute and relative paths without those prefixes are returned unchanged.
pub fn expand_dataset_path(raw: &str) -> PathBuf {
    let home = current_home_dir();
    expand_dataset_path_with_home(raw, home.as_deref())
}

pub(crate) fn expand_dataset_path_with_home(raw: &str, home: Option<&str>) -> PathBuf {
    let trimmed = raw.trim();
    let Some(home_dir) = home.filter(|h| !h.is_empty()) else {
        return PathBuf::from(trimmed);
    };

    if trimmed == "~" || trimmed == "$HOME" || trimmed.eq_ignore_ascii_case("%USERPROFILE%") {
        return PathBuf::from(home_dir);
    }

    if let Some(rest) = trimmed.strip_prefix("~/").or_else(|| trimmed.strip_prefix("~\\")) {
        return PathBuf::from(home_dir).join(rest);
    }

    if let Some(rest) = trimmed
        .strip_prefix("$HOME/")
        .or_else(|| trimmed.strip_prefix("$HOME\\"))
    {
        return PathBuf::from(home_dir).join(rest);
    }

    let userprofile_prefix = "%USERPROFILE%";
    if trimmed.len() > userprofile_prefix.len()
        && trimmed[..userprofile_prefix.len()].eq_ignore_ascii_case(userprofile_prefix)
    {
        let after = &trimmed[userprofile_prefix.len()..];
        if let Some(rest) = after.strip_prefix('/').or_else(|| after.strip_prefix('\\')) {
            return PathBuf::from(home_dir).join(rest);
        }
    }

    PathBuf::from(trimmed)
}
