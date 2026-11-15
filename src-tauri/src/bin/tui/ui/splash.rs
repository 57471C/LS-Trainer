//! Normal-buffer BTOP++-style ASCII splash and exit summary banner.

use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};

use crate::app::AppState;
use crate::ui::format_eta_seconds;

pub const SPLASH_ASCII_LINES: [&str; 8] = [
    r" /$$        /$$$$$$       /$$$$$$$$                 /$$                              ",
    r"| $$       /$$__  $$     |__  $$__/                |__/                              ",
    r"| $$      | $$  \__/        | $$  /$$$$$$  /$$$$$$  /$$ /$$$$$$$   /$$$$$$   /$$$$$$ ",
    r"| $$      |  $$$$$$  /$$$$$$| $$ /$$__  $$|____  $$| $$| $$__  $$ /$$__  $$ /$$__  $$",
    r"| $$       \____  $$|______/| $$| $$  \__/ /$$$$$$$| $$| $$  \ $$| $$$$$$$$| $$  \__/",
    r"| $$       /$$  \ $$        | $$| $$      /$$__  $$| $$| $$  | $$| $$_____/| $$      ",
    r"| $$$$$$$$|  $$$$$$/        | $$| $$     |  $$$$$$$| $$| $$  | $$|  $$$$$$$| $$      ",
    r"|________/ \______/         |__/|__/      \_______/|__/|__/  |__/ \_______/|__/      ",
];

/// Format the BTOP++-style ASCII splash for the normal terminal buffer (`stdout`).
/// Written before entering the alternate screen so it remains in terminal scrollback.
pub fn format_splash_stdout() -> String {
    let mut out = String::new();
    out.push('\n');
    for (idx, art_line) in SPLASH_ASCII_LINES.iter().enumerate() {
        let ansi_rgb = match idx {
            0..=2 => "\x1b[1;38;2;116;202;218m",
            3..=5 => "\x1b[1;38;2;68;168;186m",
            _ => "\x1b[1;38;2;62;154;174m",
        };
        out.push_str(ansi_rgb);
        out.push_str(art_line);
        out.push_str("\x1b[0m\n");
    }
    out.push('\n');
    out.push_str(
        " \x1b[1;38;2;116;202;218m── LS-Trainer TUI \x1b[0m\x1b[38;2;138;150;168m· YOLO · RF-DETR · RT-DETR · ONNX ──\x1b[0m\n\n",
    );
    out
}

pub fn write_splash_to<W: Write>(writer: &mut W) -> std::io::Result<()> {
    let splash = format_splash_stdout();
    writer.write_all(splash.as_bytes())?;
    writer.flush()
}

/// Print the BTOP++-style splash to the normal `stdout` buffer (so it stays in scrollback)
/// and wait for `duration` (~1.5–2s) or until any key is pressed before entering the TUI.
/// Returns `Ok(true)` if the user pressed `Ctrl+C` to abort immediately.
pub fn run_stdout_splash(duration: Duration) -> std::io::Result<bool> {
    let mut stdout = std::io::stdout();
    write_splash_to(&mut stdout)?;

    let raw_enabled = enable_raw_mode().is_ok();
    let deadline = Instant::now() + duration;
    let poll_step = Duration::from_millis(50);
    let mut aborted = false;

    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let wait = (deadline - now).min(poll_step);
        if event::poll(wait)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && key.code == KeyCode::Char('c')
                    {
                        aborted = true;
                    }
                    break;
                }
            }
        }
    }

    if raw_enabled {
        let _ = disable_raw_mode();
    }
    Ok(aborted)
}

pub fn format_exit_summary(state: &AppState) -> String {
    let best_metrics_str = match (state.best_map50, state.best_map50_95) {
        (Some(m50), Some(m95)) => format!(
            "mAP50: {:.4} ({:.1}%)  ·  mAP50-95: {:.4} ({:.1}%)",
            m50,
            m50 * 100.0,
            m95,
            m95 * 100.0
        ),
        (Some(m50), None) => format!("mAP50: {:.4} ({:.1}%)", m50, m50 * 100.0),
        (None, Some(m95)) => format!("mAP50-95: {:.4} ({:.1}%)", m95, m95 * 100.0),
        (None, None) => {
            if state.step_counter > 0 {
                format!(
                    "train_loss: {:.4} (epoch {}/{})",
                    state.latest_metrics.train_loss,
                    state.latest_metrics.epoch,
                    state.latest_metrics.total_epochs
                )
            } else {
                "N/A (no epochs completed)".to_string()
            }
        }
    };

    let best_weights_str = state.best_weights_path.as_deref().unwrap_or("Not saved");
    let onnx_str = state.onnx_path.as_deref().unwrap_or("Not exported");
    let run_dir_str = state
        .results_dir
        .as_deref()
        .unwrap_or("N/A (no run started)");
    let elapsed_secs = state.elapsed_seconds();
    let elapsed_str = if state.training_started_at.is_some() {
        let hms = format_eta_seconds(elapsed_secs.max(1.0));
        if hms == "--:--" {
            format!("{elapsed_secs:.1}s")
        } else {
            format!("{hms} ({elapsed_secs:.1}s)")
        }
    } else {
        "00:00".to_string()
    };

    let rows: [(&str, &str); 5] = [
        ("Best Metrics", &best_metrics_str),
        ("Best Weights", best_weights_str),
        ("ONNX Model", onnx_str),
        ("Run Directory", run_dir_str),
        ("Elapsed Time", &elapsed_str),
    ];

    let title = "─ LS-Trainer — Exit Summary ";
    let mut inner_width = 72usize;
    for (_label, val) in &rows {
        let row_len = 2 + 14 + 3 + val.chars().count() + 2;
        if row_len > inner_width {
            inner_width = row_len;
        }
    }

    let top_dashes = inner_width.saturating_sub(title.chars().count());
    let mut out = String::new();
    out.push_str(&format!("╭{title}{}╮\n", "─".repeat(top_dashes)));
    out.push_str(&format!("│{}│\n", " ".repeat(inner_width)));

    for (label, val) in &rows {
        let content = format!("  {label:<14} : {val}");
        let pad = inner_width.saturating_sub(content.chars().count());
        out.push_str(&format!("│{content}{}│\n", " ".repeat(pad)));
    }

    out.push_str(&format!("│{}│\n", " ".repeat(inner_width)));
    out.push_str(&format!("╰{}╯\n", "─".repeat(inner_width)));
    out
}

pub fn write_exit_summary_to<W: Write>(writer: &mut W, state: &AppState) -> std::io::Result<()> {
    let summary = format_exit_summary(state);
    writeln!(writer, "\x1b[38;2;116;202;218m{summary}\x1b[0m")?;
    writer.flush()
}

pub fn print_exit_summary(state: &Arc<Mutex<AppState>>) {
    if let Ok(mut guard) = state.lock() {
        guard.refresh_run_artifacts();
        let mut stdout = std::io::stdout();
        let _ = write_exit_summary_to(&mut stdout, &guard);
    }
}
