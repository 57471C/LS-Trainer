mod app;
mod event;
mod ui;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use arboard::Clipboard;
use crossterm::event as ct_event;

use crate::app::{
    launch_training_thread, parse_cli_args, spawn_hardware_polling_tasks, AppState,
};
use crate::event::{handle_event, EventOutcome};
use crate::ui::draw_ui;
use crate::ui::splash::{print_exit_summary, run_stdout_splash};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(cli_cfg) = (match parse_cli_args() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(2);
        }
    }) else {
        return Ok(());
    };

    let initial_req = if !cli_cfg.interactive {
        match cli_cfg.to_training_request() {
            Ok(req) => Some(req),
            Err(e) => {
                eprintln!("Error: {e}");
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    // Always restore terminal raw mode and alternate screen on panic
    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        ratatui::restore();
        orig_hook(panic_info);
    }));

    let state = Arc::new(Mutex::new(AppState::new(cli_cfg)));
    spawn_hardware_polling_tasks(Arc::clone(&state));

    // Write splash to normal terminal stdout BEFORE entering alternate screen so it stays in scrollback
    let aborted_in_splash = run_stdout_splash(Duration::from_millis(1650))?;
    if aborted_in_splash {
        return Ok(());
    }

    let mut terminal = ratatui::init();

    if let Some(req) = initial_req {
        launch_training_thread(Arc::clone(&state), req);
    }

    let mut clipboard: Option<Clipboard> = None;
    let tick_rate = Duration::from_millis(80);

    let show_exit_summary = loop {
        if let Ok(mut guard) = state.lock() {
            if guard.is_running && !guard.is_paused {
                guard.spinner_tick = guard.spinner_tick.wrapping_add(1);
                if guard.spinner_tick % 10 == 0 {
                    guard.refresh_run_artifacts();
                }
            }
            let _ = terminal.draw(|frame| draw_ui(frame, &guard));
        }

        if ct_event::poll(tick_rate)? {
            let ev = ct_event::read()?;
            if let EventOutcome::Quit {
                show_exit_summary: show_summary,
            } = handle_event(&state, ev, &mut clipboard)
            {
                break show_summary;
            }
        }
    };

    ratatui::restore();
    if show_exit_summary {
        print_exit_summary(&state);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use app_lib::{force_kill_active_child, GpuTelemetry, TrainMetrics};
    use crossterm::event::KeyCode;
    use ratatui::{backend::TestBackend, style::Color, Terminal};

    use crate::app::{
        parse_cli_args_from, ArchChoice, CliConfig, HealthLevel, OverlayMode, UiMode,
        BAR_EPOCH_TEAL, BAR_GPU_TEAL, BAR_TRACK_BG, GAUGE_TEXT_COVERED, GAUGE_TEXT_UNCOVERED,
        TABLE_BORDER_SOFT,
    };
    use crate::event::{handle_setup_key, handle_training_key};
    use crate::ui::splash::{
        format_exit_summary, write_exit_summary_to, write_splash_to, SPLASH_ASCII_LINES,
    };
    use crate::ui::widgets::log::parse_ansi_or_styled_log_line;

    #[test]
    fn epoch_bar_uses_muted_teal_and_vram_text_has_dark_contrast_when_covered() {
        let backend = TestBackend::new(120, 32);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        let mut cfg = CliConfig::default();
        cfg.interactive = false;
        cfg.dataset = "datasets/sample".to_string();
        let mut state = AppState::new(cfg);
        state.mode = UiMode::Training;
        state.is_running = true;
        state.apply_metrics(TrainMetrics {
            epoch: 3,
            total_epochs: 4,
            step: 8,
            total_steps: 10,
            train_loss: 1.2345,
            val_loss: Some(1.1100),
            map50: Some(0.812),
            map50_95: Some(0.584),
            lr: 1e-4,
            samples_per_sec: 24.5,
            eta_secs: 42.0,
            gpus: vec![GpuTelemetry {
                index: 0,
                name: "Intel Arc B580".to_string(),
                vram_used_gb: 12.0,
                vram_total_gb: 16.0,
                util_percent: 75.0,
            }],
        });

        terminal
            .draw(|frame| draw_ui(frame, &state))
            .expect("draw_ui succeeds");

        let buf = terminal.backend().buffer();

        // 1. Check VRAM gauge row (row y=1 inside top GPU strip):
        // With 75% fill across width ~116, the centered label "VRAM 12.00 / 16.00 GiB (75%)"
        // including "(75%)" is covered by the GPU bar and must use GAUGE_TEXT_COVERED (dark/black)
        // on BAR_GPU_TEAL background.
        let mut found_vram_pct_covered = false;
        for x in 1..118 {
            let cell = &buf[(x, 1)];
            if cell.symbol() == "%" && cell.bg == BAR_GPU_TEAL {
                assert_eq!(
                    cell.fg, GAUGE_TEXT_COVERED,
                    "VRAM percentage covered by GPU bar must be drawn in dark/black text"
                );
                found_vram_pct_covered = true;
            }
        }
        assert!(
            found_vram_pct_covered,
            "Expected VRAM percentage character '%' covered by GPU bar in row 1"
        );

        // 2. Check partial coverage (e.g. 52% where part of the label is covered and part is on track):
        state.gpus[0].vram_used_gb = 8.32;
        state.gpus[0].vram_total_gb = 16.0;
        terminal
            .draw(|frame| draw_ui(frame, &state))
            .expect("draw_ui partial vram succeeds");
        let buf = terminal.backend().buffer();
        let mut saw_covered_label_char = false;
        let mut saw_uncovered_label_char = false;
        for x in 2..118 {
            let cell = &buf[(x, 1)];
            if cell.symbol() != " " {
                if cell.bg == BAR_GPU_TEAL {
                    assert_eq!(cell.fg, GAUGE_TEXT_COVERED);
                    saw_covered_label_char = true;
                } else if cell.bg == BAR_TRACK_BG {
                    assert_eq!(cell.fg, GAUGE_TEXT_UNCOVERED);
                    saw_uncovered_label_char = true;
                }
            }
        }
        assert!(saw_covered_label_char && saw_uncovered_label_char);

        // 3. Confirm no cell in the entire buffer uses loud pink (LightMagenta) for the Epoch bar,
        // and that BAR_EPOCH_TEAL is used instead.
        let mut saw_epoch_teal = false;
        for y in 0..32 {
            for x in 0..120 {
                let cell = &buf[(x, y)];
                assert_ne!(
                    cell.bg,
                    Color::LightMagenta,
                    "Epoch bar must not use loud LightMagenta background"
                );
                if cell.bg == BAR_EPOCH_TEAL {
                    saw_epoch_teal = true;
                }
            }
        }
        assert!(
            saw_epoch_teal,
            "Expected Epoch progress bar to use BAR_EPOCH_TEAL"
        );
    }

    #[test]
    fn log_metric_tables_soften_heavy_borders() {
        let heavy_line = "┏━━━━━━━━━━━━━━┳━━━━━━━┓";
        let styled = parse_ansi_or_styled_log_line(heavy_line);
        let rendered: String = styled.spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(rendered, "╭──────────────┬───────╮");
        for span in &styled.spans {
            assert_eq!(span.style.fg, Some(TABLE_BORDER_SOFT));
        }
    }

    #[test]
    fn no_flags_opens_fullscreen_launcher_with_all_fields() {
        let empty_args: Vec<&str> = Vec::new();
        let cfg = parse_cli_args_from(empty_args)
            .expect("parse empty args")
            .expect("some config");
        assert!(
            cfg.interactive,
            "No flags must open the interactive launcher"
        );

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        {
            let guard = state.lock().unwrap();
            assert_eq!(guard.mode, UiMode::Setup);

            let backend = TestBackend::new(120, 32);
            let mut terminal = Terminal::new(backend).expect("create test terminal");
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw launcher");

            let buf = terminal.backend().buffer();
            let mut screen_text = String::new();
            for y in 0..32 {
                for x in 0..120 {
                    screen_text.push_str(buf[(x, y)].symbol());
                }
                screen_text.push('\n');
            }
            assert!(screen_text.contains("LS-Trainer TUI — Launch Menu"));
            assert!(screen_text.contains("Architecture"));
            assert!(screen_text.contains("Dataset Path"));
            assert!(screen_text.contains("Epochs"));
            assert!(screen_text.contains("Batch Size"));
            assert!(screen_text.contains("Device"));
            assert!(screen_text.contains("Workers"));
            assert!(screen_text.contains("[ Start Training (Enter / F2) ]"));
        }

        // Navigate and edit fields in the launcher:
        // Field 0: Architecture -> Right arrow switches YOLO -> RF-DETR
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::RfDetr);
        handle_setup_key(&state, KeyCode::Left);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::Yolo);

        // Down to Field 1 (Dataset Path), Enter to edit, set dataset path, Enter to confirm
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, 1);
        handle_setup_key(&state, KeyCode::Enter);
        assert!(state.lock().unwrap().setup.editing_field);
        state.lock().unwrap().setup.dataset =
            "C:\\Users\\Terry.minett\\Downloads\\PAK-BB_yolo.tar".to_string();
        handle_setup_key(&state, KeyCode::Enter);
        assert!(!state.lock().unwrap().setup.editing_field);

        // Down to Field 2 (Epochs), edit to "1"
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, 2);
        handle_setup_key(&state, KeyCode::Backspace);
        handle_setup_key(&state, KeyCode::Backspace);
        handle_setup_key(&state, KeyCode::Char('1'));
        handle_setup_key(&state, KeyCode::Enter);
        assert_eq!(state.lock().unwrap().setup.epochs, "1");

        // Down to Field 3 (Batch Size), Field 4 (Device), Field 5 (Workers)
        handle_setup_key(&state, KeyCode::Down); // 3: Batch
        handle_setup_key(&state, KeyCode::Down); // 4: Device
        handle_setup_key(&state, KeyCode::Right); // auto -> xpu
        assert_eq!(state.lock().unwrap().setup.device, "xpu");

        handle_setup_key(&state, KeyCode::Down); // 5: Workers
        handle_setup_key(&state, KeyCode::Right); // auto -> 0
        assert_eq!(state.lock().unwrap().setup.workers_input, "0");

        let req = state
            .lock()
            .unwrap()
            .setup
            .to_training_request()
            .expect("valid training request from launcher");
        assert_eq!(req.model_family, "YOLO");
        assert_eq!(req.epochs, 1);
        assert_eq!(req.batch, 4);
        assert_eq!(req.device, "xpu");
        assert_eq!(req.workers, Some(0));
    }

    #[test]
    fn with_flags_runs_immediately_without_launcher() {
        let args = vec![
            "--arch",
            "yolo",
            "--dataset",
            "C:\\Users\\Terry.minett\\Downloads\\PAK-BB_yolo.tar",
            "--epochs",
            "1",
            "--batch",
            "4",
            "--device",
            "xpu",
            "--workers",
            "0",
        ];
        let cfg = parse_cli_args_from(args)
            .expect("parse flags")
            .expect("some config");
        assert!(
            !cfg.interactive,
            "Providing training flags must start immediately without launcher"
        );
        let state = AppState::new(cfg.clone());
        assert_eq!(state.mode, UiMode::Training);
        let req = cfg.to_training_request().expect("valid request");
        assert_eq!(req.epochs, 1);
        assert_eq!(req.workers, Some(0));
    }

    fn write_tiny_bmp_64x64(path: &Path) {
        let width: u32 = 64;
        let height: u32 = 64;
        let row_bytes = width * 3;
        let pixel_bytes = row_bytes * height;
        let file_size = 54 + pixel_bytes;

        let mut data = Vec::with_capacity(file_size as usize);
        // BMP Header (14 bytes)
        data.extend_from_slice(b"BM");
        data.extend_from_slice(&file_size.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&54u32.to_le_bytes());
        // DIB Header (BITMAPINFOHEADER, 40 bytes)
        data.extend_from_slice(&40u32.to_le_bytes());
        data.extend_from_slice(&(width as i32).to_le_bytes());
        data.extend_from_slice(&(height as i32).to_le_bytes());
        data.extend_from_slice(&1u16.to_le_bytes()); // planes
        data.extend_from_slice(&24u16.to_le_bytes()); // bpp
        data.extend_from_slice(&0u32.to_le_bytes()); // compression
        data.extend_from_slice(&pixel_bytes.to_le_bytes());
        data.extend_from_slice(&2835u32.to_le_bytes());
        data.extend_from_slice(&2835u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());

        for y in 0..height {
            for x in 0..width {
                let b = ((x * 4) & 0xFF) as u8;
                let g = ((y * 4) & 0xFF) as u8;
                let r = 128u8;
                data.extend_from_slice(&[b, g, r]);
            }
        }
        std::fs::write(path, data).expect("write tiny bmp");
    }

    #[test]
    fn splash_writes_to_normal_scrollback_and_persists_after_quit() {
        let mut normal_scrollback: Vec<u8> = Vec::new();
        write_splash_to(&mut normal_scrollback).expect("write splash to normal stdout buffer");

        let splash_out = String::from_utf8(normal_scrollback.clone()).expect("utf8 splash");
        // Verify each of the 8 Big Money-ne ASCII art lines is present left-aligned in normal stdout
        for (idx, expected_line) in SPLASH_ASCII_LINES.iter().enumerate() {
            assert!(
                splash_out.contains(expected_line),
                "Splash ASCII line {idx} must be written to normal stdout scrollback"
            );
        }
        // Must not emit alternate screen or clear-screen escape codes on the normal buffer
        assert!(!splash_out.contains("\x1b[2J"));
        assert!(!splash_out.contains("\x1b[3J"));
        assert!(!splash_out.contains("\x1b[?1049h"));

        // Transition to TUI launcher on alternate screen
        let backend = TestBackend::new(100, 24);
        let mut terminal = Terminal::new(backend).expect("create test terminal");
        let state = AppState::new(CliConfig::default());
        terminal
            .draw(|frame| draw_ui(frame, &state))
            .expect("draw launcher after splash");
        let buf_after = terminal.backend().buffer();
        let mut text_after = String::new();
        for y in 0..24 {
            for x in 0..100 {
                text_after.push_str(buf_after[(x, y)].symbol());
            }
            text_after.push('\n');
        }
        assert!(text_after.contains("LS-Trainer TUI — Launch Menu"));

        // On quit (q or Ctrl+C), exit summary is written to normal stdout without clearing scrollback
        write_exit_summary_to(&mut normal_scrollback, &state)
            .expect("write exit summary to normal stdout buffer");
        let full_scrollback =
            String::from_utf8(normal_scrollback).expect("utf8 full normal scrollback");
        assert!(!full_scrollback.contains("\x1b[2J"));
        assert!(!full_scrollback.contains("\x1b[3J"));
        for expected_line in SPLASH_ASCII_LINES {
            assert!(
                full_scrollback.contains(expected_line),
                "Splash art must still be present in normal terminal scrollback after quit"
            );
        }
        assert!(
            full_scrollback.contains("╭─ LS-Trainer — Exit Summary"),
            "Exit summary must also be present in normal terminal scrollback after quit"
        );
    }

    static REAL_RUN_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn short_real_run_from_launcher_verifies_tui_rendering() {
        let _run_guard = REAL_RUN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp_ds = std::env::temp_dir().join("ls_trainer_tui_tiny_ds");
        let _ = std::fs::remove_dir_all(&tmp_ds);
        if let Ok(entries) = std::fs::read_dir("runs/detect") {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("tui-short-verify")
                {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
        let img_dir = tmp_ds.join("images").join("train");
        let lbl_dir = tmp_ds.join("labels").join("train");
        std::fs::create_dir_all(&img_dir).expect("create train images dir");
        std::fs::create_dir_all(&lbl_dir).expect("create train labels dir");

        for i in 0..2 {
            write_tiny_bmp_64x64(&img_dir.join(format!("sample_{i}.bmp")));
            std::fs::write(
                lbl_dir.join(format!("sample_{i}.txt")),
                "0 0.5 0.5 0.4 0.4\n",
            )
            .expect("write label");
        }
        let yaml_content = format!(
            "path: {}\ntrain: images/train\nval: images/train\nnames:\n  0: part\n",
            tmp_ds.to_string_lossy().replace('\\', "/")
        );
        std::fs::write(tmp_ds.join("data.yaml"), yaml_content).expect("write data.yaml");

        // 0. Print splash to normal terminal stdout buffer before starting TUI
        let mut normal_stdout_buf: Vec<u8> = Vec::new();
        write_splash_to(&mut normal_stdout_buf).expect("write splash to stdout");

        let backend = TestBackend::new(140, 40);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        // 1. Launch with no flags -> transitions from splash to Setup (Launcher) mode
        let empty_args: Vec<&str> = Vec::new();
        let cfg = parse_cli_args_from(empty_args)
            .expect("parse empty args")
            .expect("some config");
        assert!(cfg.interactive);

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        assert_eq!(state.lock().unwrap().mode, UiMode::Setup);

        // Configure 1-epoch fast XPU run in launcher and press F2 to start
        {
            let mut guard = state.lock().unwrap();
            guard.setup.arch = ArchChoice::Yolo;
            guard.setup.model = "yolo11n.pt".to_string();
            guard.setup.dataset = tmp_ds.to_string_lossy().to_string();
            guard.setup.epochs = "1".to_string();
            guard.setup.batch = "2".to_string();
            guard.setup.imgsz = 64;
            guard.setup.device = "xpu".to_string();
            guard.setup.workers_input = "0".to_string();
            guard.setup.run_name = "tui-short-verify".to_string();
        }

        // Press F2 in launcher -> transitions to UiMode::Training and spawns run_training + ONNX export
        let should_quit = handle_setup_key(&state, KeyCode::F(2));
        assert!(!should_quit);
        assert_eq!(state.lock().unwrap().mode, UiMode::Training);
        assert!(state.lock().unwrap().is_running);

        let deadline = Instant::now() + Duration::from_secs(60);
        let mut saw_live_metrics = false;
        while Instant::now() < deadline {
            let done = {
                let mut guard = state.lock().unwrap();
                if guard.is_running {
                    guard.spinner_tick = guard.spinner_tick.wrapping_add(1);
                }
                terminal
                    .draw(|frame| draw_ui(frame, &guard))
                    .expect("draw_ui during live run");
                if guard.step_counter > 0 {
                    saw_live_metrics = true;
                }
                !guard.is_running && guard.exit_code.is_some()
            };
            if done {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        {
            let guard = state.lock().unwrap();
            if guard.exit_code != Some(0) {
                for l in &guard.logs {
                    println!("LOG: {l}");
                }
                force_kill_active_child();
            }
            assert_eq!(
                guard.exit_code,
                Some(0),
                "Expected short training run + ONNX export to exit with code 0; status={}",
                guard.status_line
            );
            assert!(
                saw_live_metrics,
                "Expected live @@LS_METRICS@@ updates during run"
            );
            assert_eq!(guard.latest_metrics.epoch, 1);

            // Verify both weights (best.pt) and ONNX (best.onnx in run dir) were produced
            let run_dir = guard.results_dir.as_ref().expect("results_dir recorded");
            let best_pt = guard
                .best_weights_path
                .as_ref()
                .expect("best_weights_path recorded");
            let best_onnx = guard.onnx_path.as_ref().expect("onnx_path recorded");
            assert!(
                Path::new(best_pt).is_file(),
                "Expected best.pt weights file to exist at {best_pt}"
            );
            assert!(
                Path::new(best_onnx).is_file(),
                "Expected exported ONNX file to exist at {best_onnx}"
            );
            assert!(
                Path::new(run_dir).join("best.onnx").is_file(),
                "Expected best.onnx to be saved in run directory {run_dir}"
            );
            assert!(
                guard
                    .logs
                    .iter()
                    .any(|l| l.contains("Training complete!") && l.contains(best_onnx)),
                "Expected training complete log message to include ONNX path"
            );

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("final draw_ui");
            let buf = terminal.backend().buffer();
            let screen = buffer_to_string(buf);

            // Confirm Host / CPU panel is 12 rows tall (y=5..16, +7 rows taller than old 5-row box),
            // giving 8 vertical rows to the CPU bar chart while keeping Run Summary and Live Training Log balanced.
            let mut host_title_y = None;
            let mut run_summary_title_y = None;
            for (y, line) in screen.lines().enumerate() {
                if line.contains("Host / CPU") {
                    host_title_y = Some(y);
                }
                if line.contains("Run Summary") {
                    run_summary_title_y = Some(y);
                }
            }
            let host_h = run_summary_title_y.expect("Run Summary title visible")
                - host_title_y.expect("Host / CPU title visible");
            assert!(
                (11..=13).contains(&host_h),
                "Expected Host / CPU panel height to be 11..=13 rows (6-8 rows taller than 5), got {host_h}"
            );
            assert!(screen.contains("Telemetry Curves"));
            assert!(screen.contains("ONNX"));
            assert!(screen.contains("Live Training Log"));
            assert!(screen.contains("Stop & Quit"));

            // Confirm epoch bar uses BAR_EPOCH_TEAL and never LightMagenta, and VRAM text is readable
            let mut saw_epoch_teal = false;
            let mut saw_vram_bar = false;
            for y in 0..40 {
                for x in 0..140 {
                    let cell = &buf[(x, y)];
                    assert_ne!(cell.bg, Color::LightMagenta);
                    if cell.bg == BAR_EPOCH_TEAL {
                        saw_epoch_teal = true;
                    }
                    if y == 1 && cell.bg == BAR_GPU_TEAL && cell.symbol() != " " {
                        assert_eq!(cell.fg, GAUGE_TEXT_COVERED);
                        saw_vram_bar = true;
                    }
                }
            }
            assert!(saw_epoch_teal);
            assert!(
                !guard.gpus.is_empty() || saw_vram_bar,
                "Expected GPU telemetry during training run"
            );
        }

        // Press 'q' to quit -> verify exit summary contains border, metrics, weights path, ONNX path, run dir, and elapsed time
        let mut cb = None;
        let quit_triggered = handle_training_key(&state, KeyCode::Char('q'), &mut cb);
        assert!(quit_triggered, "Pressing q must quit training TUI");

        let guard = state.lock().unwrap();
        write_exit_summary_to(&mut normal_stdout_buf, &guard)
            .expect("write exit summary to normal stdout");
        let scrollback = String::from_utf8(normal_stdout_buf).expect("valid utf8 scrollback");
        assert!(scrollback.contains(SPLASH_ASCII_LINES[0]));
        assert!(scrollback.contains("╭─ LS-Trainer — Exit Summary"));

        let summary = format_exit_summary(&guard);
        println!("EXIT SUMMARY:\n{summary}");
        assert!(summary.contains("╭─ LS-Trainer — Exit Summary"));
        assert!(summary.contains("╰"));
        assert!(summary.contains("mAP50:"));
        assert!(summary.contains("mAP50-95:"));
        assert!(summary.contains(guard.best_weights_path.as_deref().unwrap()));
        assert!(summary.contains(guard.onnx_path.as_deref().unwrap()));
        assert!(summary.contains(guard.results_dir.as_deref().unwrap()));
        assert!(summary.contains("Elapsed Time"));

        let _ = std::fs::remove_dir_all(&tmp_ds);
        if let Ok(entries) = std::fs::read_dir("runs/detect") {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("tui-short-verify")
                {
                    let _ = std::fs::remove_dir_all(entry.path());
                }
            }
        }
    }

    fn buffer_to_string(buf: &ratatui::buffer::Buffer) -> String {
        let area = buf.area;
        let mut out = String::new();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                out.push_str(buf[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn dual_xpu_ui_paths_overall_progress_new_best_health_and_overlays() {
        let backend = TestBackend::new(140, 34);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        let args = vec![
            "--arch",
            "rf-detr",
            "--dataset",
            "datasets/coco_parts",
            "--epochs",
            "4",
            "--batch",
            "4",
            "--device",
            "xpu:0,1",
            "--workers",
            "0",
            "--run-name",
            "rfdetr-dual-xpu-test",
        ];
        let cfg = parse_cli_args_from(args)
            .expect("parse flags")
            .expect("some config");
        let req = cfg.to_training_request().expect("valid request");

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        {
            let mut guard = state.lock().unwrap();
            guard.mode = UiMode::Training;
            guard.active_request = Some(req);
            guard.is_running = true;
            guard.dataset_train_images = Some(80);
            guard.results_dir =
                Some(r"c:\Scripts\LS-Trainer\runs\detect\rfdetr-dual-xpu-test".to_string());
        }

        // 1. Verify Healthy status initially & dual-XPU telemetry + Overall progress + New-best indicator
        {
            let mut guard = state.lock().unwrap();
            guard.apply_metrics(TrainMetrics {
                epoch: 2,
                total_epochs: 4,
                step: 5,
                total_steps: 10,
                train_loss: 1.85,
                val_loss: Some(1.62),
                map50: Some(0.645),
                map50_95: Some(0.412),
                lr: 1e-4,
                samples_per_sec: 18.4,
                eta_secs: 35.0,
                gpus: vec![
                    GpuTelemetry {
                        index: 0,
                        name: "Intel Arc B580".to_string(),
                        vram_used_gb: 9.2,
                        vram_total_gb: 12.0,
                        util_percent: 82.0,
                    },
                    GpuTelemetry {
                        index: 1,
                        name: "Intel Arc B580".to_string(),
                        vram_used_gb: 8.9,
                        vram_total_gb: 12.0,
                        util_percent: 79.0,
                    },
                ],
            });
            guard.best_weights_path = Some(
                r"c:\Scripts\LS-Trainer\runs\detect\rfdetr-dual-xpu-test\checkpoint_best_total.pth"
                    .to_string(),
            );
            guard.onnx_path = Some(
                r"c:\Scripts\LS-Trainer\runs\detect\rfdetr-dual-xpu-test\best.onnx".to_string(),
            );

            assert!(
                guard.is_new_best_active(),
                "New-best indicator must fire when mAP/loss improves"
            );
            assert_eq!(guard.health_level, HealthLevel::Healthy);

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw dual-xpu training ui");
            let screen = buffer_to_string(terminal.backend().buffer());

            // Dual-XPU cards rendered side-by-side
            assert!(screen.contains("GPU 0: Intel Arc B580"));
            assert!(screen.contains("GPU 1: Intel Arc B580"));
            // Overall progress bar with combined epoch + step + images
            assert!(
                screen.contains("Overall 38% · Ep 2/4 · Step 5/10 · 120/320 imgs"),
                "Expected Overall progress indicator in screen:\n{screen}"
            );
            // New-best indicator badge next to Best mAP
            assert!(
                screen.contains("★ New best"),
                "Expected New best marker in Run Summary:\n{screen}"
            );
            // Model output paths in Run Summary during the run
            assert!(screen.contains("Run Dir"));
            assert!(screen.contains("Best Wts"));
            assert!(screen.contains("ONNX"));
            assert!(screen.contains("checkpoint_best_total.pth"));
            assert!(screen.contains("best.onnx"));
            // Healthy status line
            assert!(screen.contains("● Healthy"));
        }

        // 2. Verify noisy warnings (triton, sync_dist) update Status Line to Warning and are suppressed in live log
        {
            let mut guard = state.lock().unwrap();
            let log_len_before = guard.logs.len();
            guard.push_log(
                "W1115 16:00:00.000000 1234 torch/utils/flop_counter.py:45] triton not found; flop counting will not work"
                    .to_string(),
            );
            guard.push_log(
                "PossibleUserWarning: It is recommended to use `self.log('val/loss', ..., sync_dist=True)`"
                    .to_string(),
            );
            assert_eq!(
                guard.logs.len(),
                log_len_before,
                "Noisy triton and sync_dist warnings must be suppressed from flooding live logs"
            );
            assert_eq!(guard.health_level, HealthLevel::Warning);
            assert_eq!(guard.suppressed_warning_count, 2);

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw warning state");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("▲ Warning"));
            assert!(screen.contains("triton, sync_dist"));
            assert!(screen.contains("2 suppressed in log"));
        }

        // 3. Verify F1 / ? Keyboard Shortcuts Help overlay and F3 Config Snapshot overlay and F4 History overlay
        let mut cb = None;
        handle_training_key(&state, KeyCode::Char('?'), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Help);
        {
            let guard = state.lock().unwrap();
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw help overlay");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Keyboard Shortcuts Help"));
            assert!(screen.contains("Toggle Run Configuration Snapshot overlay"));
        }
        handle_training_key(&state, KeyCode::Esc, &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        handle_training_key(&state, KeyCode::F(3), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Config);
        {
            let guard = state.lock().unwrap();
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw config overlay");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Run Configuration Snapshot"));
            assert!(screen.contains("RF-DETR"));
            assert!(screen.contains("xpu:0,1"));
            assert!(screen.contains("datasets/coco_parts"));
        }
        handle_training_key(&state, KeyCode::F(3), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        handle_training_key(&state, KeyCode::F(4), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::History);
        {
            let guard = state.lock().unwrap();
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw history overlay");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Recent Training Runs"));
        }
        handle_training_key(&state, KeyCode::Esc, &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);
    }

    #[test]
    fn short_dual_xpu_rfdetr_real_run_verifies_live_ui_and_onnx() {
        let _run_guard = REAL_RUN_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tmp_coco = std::env::temp_dir().join("ls_trainer_tui_rfdetr_coco_ds");
        let _ = std::fs::remove_dir_all(&tmp_coco);
        let run_name = "tui-rfdetr-dual-xpu-verify";
        let _ = std::fs::remove_dir_all(PathBuf::from("runs/detect").join(run_name));

        for split in ["train", "valid", "test"] {
            let split_dir = tmp_coco.join(split);
            std::fs::create_dir_all(&split_dir).expect("create coco split dir");
            for i in 0..2 {
                write_tiny_bmp_64x64(&split_dir.join(format!("img_{i}.bmp")));
            }
            let coco_json = serde_json::json!({
                "info": { "description": "tiny rfdetr verify", "version": "1.0" },
                "licenses": [],
                "categories": [
                    { "id": 0, "name": "part", "supercategory": "defect" }
                ],
                "images": [
                    { "id": 1, "file_name": "img_0.bmp", "width": 64, "height": 64 },
                    { "id": 2, "file_name": "img_1.bmp", "width": 64, "height": 64 }
                ],
                "annotations": [
                    { "id": 1, "image_id": 1, "category_id": 0, "bbox": [12.0, 12.0, 32.0, 32.0], "area": 1024.0, "iscrowd": 0 },
                    { "id": 2, "image_id": 2, "category_id": 0, "bbox": [10.0, 10.0, 36.0, 36.0], "area": 1296.0, "iscrowd": 0 }
                ]
            });
            std::fs::write(
                split_dir.join("_annotations.coco.json"),
                serde_json::to_string_pretty(&coco_json).unwrap(),
            )
            .expect("write _annotations.coco.json");
        }

        let args = vec![
            "--arch".to_string(),
            "rf-detr".to_string(),
            "--dataset".to_string(),
            tmp_coco.to_string_lossy().to_string(),
            "--epochs".to_string(),
            "1".to_string(),
            "--batch".to_string(),
            "2".to_string(),
            "--imgsz".to_string(),
            "224".to_string(),
            "--device".to_string(),
            "xpu:0,1".to_string(),
            "--workers".to_string(),
            "0".to_string(),
            "--run-name".to_string(),
            run_name.to_string(),
        ];
        let cfg = parse_cli_args_from(args)
            .expect("parse rf-detr flags")
            .expect("some config");
        let req = cfg.to_training_request().expect("valid rf-detr request");

        let state = Arc::new(Mutex::new(AppState::new(cfg)));
        let backend = TestBackend::new(140, 34);
        let mut terminal = Terminal::new(backend).expect("create test terminal");

        launch_training_thread(Arc::clone(&state), req);

        // Verify Run Dir path is immediately visible in UI while run is starting
        {
            let guard = state.lock().unwrap();
            assert!(guard.results_dir.is_some());
            assert_eq!(guard.dataset_train_images, Some(2));
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("initial draw");
            let screen = buffer_to_string(terminal.backend().buffer());
            assert!(screen.contains("Run Dir"));
            assert!(screen.contains(run_name));
            assert!(screen.contains("Overall"));
        }

        // Toggle F1 (?) and F3 (Config Snapshot) during live run
        let mut cb = None;
        handle_training_key(&state, KeyCode::F(1), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Help);
        handle_training_key(&state, KeyCode::F(1), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        handle_training_key(&state, KeyCode::F(3), &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::Config);
        handle_training_key(&state, KeyCode::Esc, &mut cb);
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        let deadline = Instant::now() + Duration::from_secs(180);
        let mut saw_live_weights_during_run = false;
        let mut saw_new_best_during_run = false;
        while Instant::now() < deadline {
            let done = {
                let mut guard = state.lock().unwrap();
                if guard.is_running {
                    guard.spinner_tick = guard.spinner_tick.wrapping_add(1);
                    guard.refresh_run_artifacts();
                    if guard.best_weights_path.is_some() {
                        saw_live_weights_during_run = true;
                    }
                    if guard.is_new_best_active() {
                        saw_new_best_during_run = true;
                    }
                }
                terminal
                    .draw(|frame| draw_ui(frame, &guard))
                    .expect("draw_ui during rf-detr run");
                !guard.is_running && guard.exit_code.is_some()
            };
            if done {
                break;
            }
            std::thread::sleep(Duration::from_millis(120));
        }

        {
            let guard = state.lock().unwrap();
            if guard.exit_code != Some(0) {
                for l in &guard.logs {
                    println!("RFDETR LOG: {l}");
                }
                force_kill_active_child();
            }
            assert_eq!(
                guard.exit_code,
                Some(0),
                "Expected short RF-DETR run + ONNX export to succeed; status={}",
                guard.status_line
            );
            assert!(
                saw_live_weights_during_run || guard.best_weights_path.is_some(),
                "Expected best weights path to be populated"
            );
            assert!(
                saw_new_best_during_run || guard.new_best_until.is_some(),
                "Expected new-best indicator to fire during RF-DETR run"
            );
            let best_ckpt = guard
                .best_weights_path
                .as_ref()
                .expect("RF-DETR best checkpoint recorded");
            let best_onnx = guard
                .onnx_path
                .as_ref()
                .expect("RF-DETR ONNX path recorded");
            assert!(
                Path::new(best_ckpt).is_file(),
                "RF-DETR checkpoint file must exist at {best_ckpt}"
            );
            assert!(
                Path::new(best_onnx).is_file(),
                "RF-DETR exported ONNX file must exist at {best_onnx}"
            );

            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("final draw_ui after rf-detr run");
            let final_screen = buffer_to_string(terminal.backend().buffer());
            assert!(final_screen.contains("Overall 100%"));
            assert!(final_screen.contains("best.onnx"));
            assert!(
                final_screen.contains("● Healthy") || final_screen.contains("▲ Warning"),
                "Status line must reflect Healthy or Warning state"
            );
        }

        let _ = std::fs::remove_dir_all(&tmp_coco);
        let _ = std::fs::remove_dir_all(PathBuf::from("runs/detect").join(run_name));
    }
}
