mod app;
mod event;
mod ui;

use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arboard::Clipboard;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::event as ct_event;
use crossterm::execute;

struct MouseCapture;

impl MouseCapture {
    fn enable() -> Self {
        let _ = execute!(io::stdout(), EnableMouseCapture);
        let _ = io::stdout().flush();
        Self
    }
}

impl Drop for MouseCapture {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), DisableMouseCapture);
        let _ = io::stdout().flush();
    }
}

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
    let _mouse = MouseCapture::enable();

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
            guard.tick_multi_xpu_metrics();
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

    drop(_mouse);
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
        parse_cli_args_from, yolo_multi_xpu_request, ArchChoice, BrowseEntryKind, BrowseKind,
        CliConfig, HealthLevel, OverlayMode, UiMode, BAR_EPOCH_TEAL, BAR_GPU_TEAL, BAR_TRACK_BG,
        BROWSE_DIR_FG,
        FIELD_AUG_PRESET, FIELD_BATCH, FIELD_DEVICE, FIELD_EPOCHS, FIELD_FREEZE, FIELD_IMGSZ,
        FIELD_RESUME, FIELD_RUN_NAME, FIELD_WORKERS, GAUGE_TEXT_COVERED, GAUGE_TEXT_UNCOVERED,
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

        // Verify webview-parity defaults on no-flag launch:
        assert_eq!(cfg.arch, ArchChoice::Yolo);
        assert_eq!(cfg.model, "yolo11m.pt"); // medium default matching webview
        assert_eq!(cfg.epochs, "60");
        assert_eq!(cfg.batch, "4");
        assert_eq!(cfg.imgsz, 960);
        assert_eq!(cfg.imgsz_input, "960");
        assert_eq!(cfg.workers, None);
        assert_eq!(cfg.workers_input, "auto");
        assert_eq!(cfg.device, "auto");
        assert!(!cfg.freeze_backbone);
        assert_eq!(cfg.freeze_layers, 10);
        assert!(cfg.amp);
        assert_eq!(cfg.dataset, "~/Exports/");
        assert!(!cfg.run_name_custom);
        assert_eq!(
            cfg.run_name,
            format!("Y11m-e60-960_{}", crate::app::local_date_dd_mm_yy())
        );
        assert_eq!(cfg.active_aug_preset(), "bench");
        assert_eq!(cfg.degrees, "180");
        assert_eq!(cfg.flipud, "0.5");
        assert_eq!(cfg.scale, "0.05");
        assert_eq!(cfg.translate, "0.1");
        assert_eq!(cfg.mosaic, "0.0");
        assert_eq!(cfg.close_mosaic, "10");
        assert_eq!(cfg.mixup, "0.0");
        assert!(!cfg.copy_paste);

        // Verify RF-DETR model scale list matches webview (nano, small, medium [default], large, base, seg-*):
        let rfdetr_scales: Vec<&str> = ArchChoice::RfDetr
            .model_options()
            .iter()
            .map(|o| o.scale)
            .collect();
        assert_eq!(
            rfdetr_scales,
            vec![
                "nano",
                "small",
                "medium",
                "large",
                "base",
                "seg-nano",
                "seg-small",
                "seg-medium",
                "seg-large",
                "seg-preview",
            ]
        );
        assert_eq!(ArchChoice::RfDetr.default_model(), "rf-detr-medium");

        // YOLO11 and YOLO26 are separate selector choices. Each keeps n/s/m/l/x.
        let yolo11_ids: Vec<&str> = ArchChoice::Yolo
            .model_options()
            .iter()
            .map(|o| o.id)
            .collect();
        assert_eq!(
            yolo11_ids,
            vec![
                "yolo11n.pt",
                "yolo11s.pt",
                "yolo11m.pt",
                "yolo11l.pt",
                "yolo11x.pt",
                "yolo11s-seg.pt",
                "yolo11m-seg.pt",
            ]
        );
        assert_eq!(ArchChoice::Yolo.default_model(), "yolo11m.pt");
        assert_eq!(ArchChoice::Yolo.display_label(), "YOLO11");
        let yolo26_ids: Vec<&str> = ArchChoice::Yolo26
            .model_options()
            .iter()
            .map(|o| o.id)
            .collect();
        assert_eq!(
            yolo26_ids,
            vec![
                "yolo26n.pt",
                "yolo26s.pt",
                "yolo26m.pt",
                "yolo26l.pt",
                "yolo26x.pt",
                "yolo26s-seg.pt",
            ]
        );
        assert_eq!(ArchChoice::Yolo26.default_model(), "yolo26m.pt");
        assert_eq!(ArchChoice::Yolo26.display_label(), "YOLO26");

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
            assert!(screen_text.contains("YOLO11"));
            assert!(screen_text.contains("yolo11"));
            assert!(screen_text.contains("~/Exports/"));
            assert!(screen_text.contains("Y11m-e60-960_"));
            assert!(screen_text.contains("Architecture"));
            assert!(screen_text.contains("Model Scale"));
            assert!(screen_text.contains("Dataset Path"));
            assert!(screen_text.contains("Resume from"));
            assert!(screen_text.contains("<fresh start>"));
            assert!(!screen_text.contains("Resume active:"));
            assert!(screen_text.contains("Epochs"));
            assert!(screen_text.contains("Batch Size"));
            assert!(screen_text.contains("Resolution"));
            assert!(screen_text.contains("Workers"));
            assert!(screen_text.contains("Device"));
            assert!(screen_text.contains("Freeze Backbone"));
            assert!(screen_text.contains("Mixed Prec (AMP)"));
            assert!(screen_text.contains("Run Name"));
            assert!(screen_text.contains("Aug Preset"));
            assert!(screen_text.contains("Degrees"));
            assert!(screen_text.contains("Flip UD"));
            assert!(screen_text.contains("Scale"));
            assert!(screen_text.contains("Translate"));
            assert!(screen_text.contains("Mosaic"));
            assert!(screen_text.contains("Close Mosaic"));
            assert!(screen_text.contains("MixUp"));
            assert!(screen_text.contains("Copy-Paste"));
            assert!(screen_text.contains("[ Start Training (Enter / F2) ]"));
            assert!(screen_text.contains("Version v0.2.2"));
            // Verify Version v0.2.2 is on the very bottom left in yellow
            let version_row = 31u16;
            assert_eq!(buf[(1, version_row)].symbol(), "V");
            assert_eq!(buf[(1, version_row)].fg, Color::Yellow);
        }

        // Navigate and edit fields in the launcher:
        // Field 0: Architecture -> Right from YOLO11 (medium) selects YOLO26 (yolo26m.pt),
        // then RF-DETR (rf-detr-medium). The size stays medium across those families.
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::Yolo26);
        assert_eq!(state.lock().unwrap().setup.model, "yolo26m.pt");
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::RfDetr);
        assert_eq!(state.lock().unwrap().setup.model, "rf-detr-medium");

        // Down to Field 1 (Model Scale): Left arrow cycles medium -> small (rf-detr-small), Right arrow -> medium -> large
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, 1);
        handle_setup_key(&state, KeyCode::Left);
        assert_eq!(state.lock().unwrap().setup.model, "rf-detr-small");
        handle_setup_key(&state, KeyCode::Left);
        assert_eq!(state.lock().unwrap().setup.model, "rf-detr-nano");
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.model, "rf-detr-small");

        // Switch back to Field 0. Left from RF-DETR (small) selects YOLO26 (yolo26s.pt),
        // then YOLO11 (yolo11s.pt).
        handle_setup_key(&state, KeyCode::Up);
        assert_eq!(state.lock().unwrap().setup.focused_field, 0);
        handle_setup_key(&state, KeyCode::Left);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::Yolo26);
        assert_eq!(state.lock().unwrap().setup.model, "yolo26s.pt");
        handle_setup_key(&state, KeyCode::Left);
        assert_eq!(state.lock().unwrap().setup.arch, ArchChoice::Yolo);
        assert_eq!(state.lock().unwrap().setup.model, "yolo11s.pt");

        // Down to Field 1 (Model Scale), Right arrow cycles small -> medium (yolo11m.pt) -> large (yolo11l.pt)
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, 1);
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.model, "yolo11m.pt");
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.model, "yolo11l.pt");

        // Down to Field 2 (Dataset Path), Enter to edit, set dataset path, Enter to confirm
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, 2);
        handle_setup_key(&state, KeyCode::Enter);
        assert!(state.lock().unwrap().setup.editing_field);
        state.lock().unwrap().setup.dataset =
            "C:\\Users\\Terry.minett\\Downloads\\PAK-BB_yolo.tar".to_string();
        handle_setup_key(&state, KeyCode::Enter);
        assert!(!state.lock().unwrap().setup.editing_field);

        // Down to Resume from. Blank keeps a fresh start.
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_RESUME);
        assert!(state.lock().unwrap().setup.resume_from.is_empty());

        // Down to Epochs, edit to "15"
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_EPOCHS);
        handle_setup_key(&state, KeyCode::Backspace);
        handle_setup_key(&state, KeyCode::Backspace);
        handle_setup_key(&state, KeyCode::Char('1'));
        handle_setup_key(&state, KeyCode::Char('5'));
        handle_setup_key(&state, KeyCode::Enter);
        assert_eq!(state.lock().unwrap().setup.epochs, "15");

        // Down to Batch Size, cycle Right 4 -> 8
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_BATCH);
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.batch, "8");

        // Down to Resolution / imgsz, cycle Right 960 -> 1280
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_IMGSZ);
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.imgsz_input, "1280");
        assert_eq!(state.lock().unwrap().setup.imgsz, 1280);

        // Down to Workers, cycle Right auto -> 0
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_WORKERS);
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.workers_input, "0");

        // Down to Device, cycle Right auto -> xpu
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_DEVICE);
        handle_setup_key(&state, KeyCode::Right);
        assert_eq!(state.lock().unwrap().setup.device, "xpu");

        // Down to Freeze Backbone, toggle false -> true
        handle_setup_key(&state, KeyCode::Down);
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_FREEZE);
        handle_setup_key(&state, KeyCode::Right);
        assert!(state.lock().unwrap().setup.freeze_backbone);

        // Down through AMP and Run Name to Aug Preset: cycle bench -> off -> light
        handle_setup_key(&state, KeyCode::Down); // AMP
        handle_setup_key(&state, KeyCode::Down); // Run Name
        handle_setup_key(&state, KeyCode::Down); // Aug Preset
        assert_eq!(state.lock().unwrap().setup.focused_field, FIELD_AUG_PRESET);
        handle_setup_key(&state, KeyCode::Right); // bench -> off
        assert_eq!(state.lock().unwrap().setup.active_aug_preset(), "off");
        assert_eq!(state.lock().unwrap().setup.degrees, "0");
        handle_setup_key(&state, KeyCode::Left); // off -> bench
        assert_eq!(state.lock().unwrap().setup.active_aug_preset(), "bench");
        assert_eq!(state.lock().unwrap().setup.degrees, "180");

        let req = state
            .lock()
            .unwrap()
            .setup
            .to_training_request()
            .expect("valid training request from launcher");
        assert_eq!(req.model_family, "YOLO");
        assert_eq!(req.model_choice, "yolo11l.pt");
        assert_eq!(req.epochs, 15);
        assert_eq!(req.batch, 8);
        assert_eq!(req.imgsz, 1280);
        assert_eq!(req.device, "xpu");
        assert_eq!(req.workers, Some(0));
        assert!(req.freeze_backbone);
        assert_eq!(req.freeze_layers, 10);
        assert!(req.amp);
        assert_eq!(req.degrees, 180.0);
        assert_eq!(req.flipud, 0.5);
        assert_eq!(req.scale, 0.05);
        assert_eq!(req.translate, 0.1);
        assert_eq!(req.mosaic, 0.0);
        assert_eq!(req.close_mosaic, 10);
        assert!(req.resume_from.is_empty());
        assert!(!req.run_name.ends_with("-ft"));
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

    #[test]
    fn yolo11_and_yolo26_sizes_select_the_matching_weights() {
        let yolo11 = parse_cli_args_from(["--arch", "yolo", "--scale", "s", "--dataset", "data"])
            .expect("parse yolo")
            .expect("config");
        assert_eq!(yolo11.arch, ArchChoice::Yolo);
        assert_eq!(yolo11.model, "yolo11s.pt");
        assert_eq!(
            yolo11.to_training_request().expect("request").model_family,
            "YOLO"
        );
        assert_eq!(
            yolo11.to_training_request().expect("request").model_choice,
            "yolo11s.pt"
        );

        let yolo11_named =
            parse_cli_args_from(["--arch", "yolo11", "--scale", "x", "--dataset", "data"])
                .expect("parse yolo11")
                .expect("config");
        assert_eq!(yolo11_named.model, "yolo11x.pt");

        for (scale, expected) in [
            ("n", "yolo26n.pt"),
            ("s", "yolo26s.pt"),
            ("m", "yolo26m.pt"),
            ("l", "yolo26l.pt"),
            ("x", "yolo26x.pt"),
            ("v26-small", "yolo26s.pt"),
        ] {
            let cfg = parse_cli_args_from(["--arch", "yolo26", "--scale", scale, "--dataset", "data"])
                .expect("parse yolo26")
                .expect("config");
            assert_eq!(cfg.arch, ArchChoice::Yolo26);
            assert_eq!(cfg.model, expected, "scale {scale}");
            let req = cfg.to_training_request().expect("request");
            assert_eq!(req.model_family, "YOLO");
            assert_eq!(req.model_choice, expected);
        }
    }

    #[test]
    fn suggested_run_name_uses_family_scale_epochs_size_and_date() {
        let mut cfg = CliConfig::default();
        cfg.arch = ArchChoice::RfDetr;
        cfg.model = "rf-detr-medium".to_string();
        cfg.epochs = "60".to_string();
        cfg.imgsz = 960;
        cfg.imgsz_input = "960".to_string();
        assert_eq!(
            cfg.suggested_run_name_on("17-11-26"),
            "RF-Dm-e60-960_17-11-26"
        );

        cfg.arch = ArchChoice::RtDetr;
        cfg.model = "rtdetr-v2-r18".to_string();
        cfg.epochs = "2".to_string();
        cfg.imgsz = 1280;
        cfg.imgsz_input = "1280".to_string();
        assert_eq!(
            cfg.suggested_run_name_on("17-11-26"),
            "RT-Dr18-e2-1280_17-11-26"
        );

        cfg.arch = ArchChoice::Yolo26;
        cfg.model = "yolo26m.pt".to_string();
        cfg.epochs = "100".to_string();
        assert_eq!(
            cfg.suggested_run_name_on("17-11-26"),
            "Y26m-e100-1280_17-11-26"
        );

        cfg.arch = ArchChoice::Yolo;
        cfg.model = "yolo11l.pt".to_string();
        cfg.epochs = "100".to_string();
        cfg.imgsz = 640;
        cfg.imgsz_input = "640".to_string();
        assert_eq!(
            cfg.suggested_run_name_on("17-11-26"),
            "Y11l-e100-640_17-11-26"
        );

        let dated = parse_cli_args_from([
            "--arch",
            "yolo26",
            "--epochs",
            "100",
            "--imgsz",
            "1280",
            "--dataset",
            "data",
        ])
        .expect("parse")
        .expect("config");
        assert!(dated.run_name.starts_with("Y26m-e100-1280_"));
        assert!(!dated.run_name_custom);

        let named = parse_cli_args_from([
            "--arch",
            "yolo26",
            "--epochs",
            "100",
            "--run-name",
            "kept-name",
            "--dataset",
            "data",
        ])
        .expect("parse")
        .expect("config");
        assert_eq!(named.run_name, "kept-name");
        assert!(named.run_name_custom);

        #[cfg(unix)]
        {
            let out = std::process::Command::new("date")
                .arg("+%d-%m-%y")
                .output()
                .expect("date");
            let expected = String::from_utf8(out.stdout)
                .expect("utf8")
                .trim()
                .to_string();
            assert_eq!(crate::app::local_date_dd_mm_yy(), expected);
        }
    }

    #[test]
    fn run_name_tracks_epochs_until_the_user_edits_it() {
        let state = Arc::new(Mutex::new(AppState::new(CliConfig::default())));
        {
            let mut guard = state.lock().unwrap();
            guard.setup.focused_field = FIELD_EPOCHS;
        }
        handle_setup_key(&state, KeyCode::Right);
        {
            let guard = state.lock().unwrap();
            assert_eq!(guard.setup.epochs, "100");
            assert!(guard.setup.run_name.contains("-e100-"));
            assert!(!guard.setup.run_name_custom);
        }

        {
            let mut guard = state.lock().unwrap();
            guard.setup.focused_field = FIELD_RUN_NAME;
        }
        handle_setup_key(&state, KeyCode::Char('X'));
        handle_setup_key(&state, KeyCode::Enter);
        let custom = state.lock().unwrap().setup.run_name.clone();
        assert!(custom.ends_with('X'), "{custom}");
        assert!(state.lock().unwrap().setup.run_name_custom);

        {
            let mut guard = state.lock().unwrap();
            guard.setup.focused_field = FIELD_EPOCHS;
        }
        handle_setup_key(&state, KeyCode::Right);
        {
            let guard = state.lock().unwrap();
            assert_eq!(guard.setup.epochs, "150");
            assert_eq!(guard.setup.run_name, custom);
        }
    }

    #[test]
    fn resume_from_fine_tunes_and_suffixes_only_the_suggested_name() {
        let dir = std::env::temp_dir().join(format!(
            "ls_trainer_resume_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let yolo_pt = dir.join("best.pt");
        let detr_pth = dir.join("checkpoint_best_total.pth");
        let wrong = dir.join("notes.txt");
        std::fs::write(&yolo_pt, b"pt").unwrap();
        std::fs::write(&detr_pth, b"pth").unwrap();
        std::fs::write(&wrong, b"nope").unwrap();

        let fresh = CliConfig::default();
        let fresh_req = fresh.to_training_request().expect("fresh request");
        assert!(fresh_req.resume_from.is_empty());
        assert_eq!(fresh_req.model_choice, "yolo11m.pt");
        assert!(!fresh_req.run_name.ends_with("-ft"));
        assert_eq!(fresh.suggested_run_name_on("17-11-26"), "Y11m-e60-960_17-11-26");

        let mut yolo = CliConfig::default();
        yolo.dataset = dir.to_string_lossy().to_string();
        yolo.resume_from = yolo_pt.to_string_lossy().to_string();
        yolo.refresh_suggested_run_name();
        assert_eq!(
            yolo.suggested_run_name_on("17-11-26"),
            "Y11m-e60-960_17-11-26-ft"
        );
        let yolo_req = yolo.to_training_request().expect("yolo resume");
        assert_eq!(yolo_req.model_choice, "yolo11m.pt");
        assert!(yolo_req.resume_from.ends_with("best.pt"));
        assert!(yolo_req.run_name.ends_with("-ft"));
        assert_eq!(yolo_req.starting_weights(), yolo_req.resume_from.as_str());

        yolo.run_name = "kept-name".to_string();
        yolo.run_name_custom = true;
        yolo.refresh_suggested_run_name();
        assert_eq!(yolo.run_name, "kept-name");
        let kept = yolo.to_training_request().expect("custom name");
        assert_eq!(kept.run_name, "kept-name");
        assert!(kept.resume_from.ends_with("best.pt"));

        let mut missing = CliConfig::default();
        missing.dataset = dir.to_string_lossy().to_string();
        missing.resume_from = dir.join("missing.pt").to_string_lossy().to_string();
        let err = missing.to_training_request().expect_err("missing file");
        assert!(err.contains("not found"), "{err}");

        let mut bad_ext = CliConfig::default();
        bad_ext.dataset = dir.to_string_lossy().to_string();
        bad_ext.resume_from = detr_pth.to_string_lossy().to_string();
        let err = bad_ext.to_training_request().expect_err("pth on yolo");
        assert!(err.contains("model="), "{err}");

        let mut rf = CliConfig::default();
        rf.arch = ArchChoice::RfDetr;
        rf.model = "rf-detr-medium".to_string();
        rf.dataset = dir.to_string_lossy().to_string();
        rf.resume_from = detr_pth.to_string_lossy().to_string();
        rf.refresh_suggested_run_name();
        let rf_req = rf.to_training_request().expect("rf resume");
        assert_eq!(rf_req.model_choice, "rf-detr-medium");
        assert!(rf_req.resume_from.ends_with("checkpoint_best_total.pth"));
        assert!(rf_req.run_name.starts_with("RF-Dm-e60-960_"));
        assert!(rf_req.run_name.ends_with("-ft"));

        let mut rt = CliConfig::default();
        rt.arch = ArchChoice::RtDetr;
        rt.model = "rtdetr-v2-r18".to_string();
        rt.dataset = dir.to_string_lossy().to_string();
        rt.resume_from = wrong.to_string_lossy().to_string();
        let err = rt.to_training_request().expect_err("txt on rt-detr");
        assert!(err.contains("pretrain_weights"), "{err}");

        let state = Arc::new(Mutex::new(AppState::new(CliConfig::default())));
        {
            let mut guard = state.lock().unwrap();
            guard.setup.focused_field = FIELD_RESUME;
            guard.setup.dataset = dir.to_string_lossy().to_string();
        }
        for ch in yolo_pt.to_string_lossy().chars() {
            handle_setup_key(&state, KeyCode::Char(ch));
        }
        handle_setup_key(&state, KeyCode::Enter);
        let typed = state.lock().unwrap().setup.to_training_request().unwrap();
        assert_eq!(typed.model_choice, "yolo11m.pt");
        assert!(typed.resume_from.ends_with("best.pt"));
        assert!(typed.run_name.ends_with("-ft"));
        {
            let guard = state.lock().unwrap();
            let backend = TestBackend::new(140, 40);
            let mut terminal = Terminal::new(backend).expect("terminal");
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw resume status");
            let buf = terminal.backend().buffer();
            let mut screen = String::new();
            for y in 0..40 {
                for x in 0..140 {
                    screen.push_str(buf[(x, y)].symbol());
                }
                screen.push('\n');
            }
            assert!(screen.contains("Resume active:"), "{screen}");
            assert!(screen.contains("best.pt"), "{screen}");
            assert!(screen.contains("model="), "{screen}");
            assert!(screen.contains("-ft"), "{screen}");
        }

        let weights_dir = dir.join("weights");
        let nested = weights_dir.join("nested");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(weights_dir.join("a.pt"), b"a").unwrap();
        std::fs::write(weights_dir.join("b.pt"), b"b").unwrap();
        std::fs::write(weights_dir.join("skip.txt"), b"x").unwrap();
        std::fs::write(nested.join("c.pt"), b"c").unwrap();
        std::fs::write(dir.join("data.tar"), b"not-a-real-tar").unwrap();
        let listed = crate::app::list_browse_dir(BrowseKind::Weights, &weights_dir);
        assert_eq!(listed[0].label, "..");
        assert_eq!(listed[0].kind, BrowseEntryKind::Parent);
        assert!(listed.iter().any(|entry| entry.label == "nested/"
            && entry.kind == BrowseEntryKind::Dir));
        assert!(listed.iter().any(|entry| entry.label == "a.pt"));
        assert!(listed.iter().any(|entry| entry.label == "b.pt"));
        assert!(listed.iter().all(|entry| entry.label != "skip.txt"));
        assert!(listed.iter().all(|entry| entry.label != "c.pt"));

        let archives = crate::app::list_browse_dir(BrowseKind::Dataset, &dir);
        assert!(archives.iter().any(|entry| entry.label == "data.tar"
            && entry.kind == BrowseEntryKind::File));
        assert!(archives
            .iter()
            .any(|entry| entry.label == "weights/" && entry.kind == BrowseEntryKind::Dir));
        assert!(archives.iter().all(|entry| entry.kind != BrowseEntryKind::File
            || entry.label.ends_with(".tar")));

        {
            let mut guard = state.lock().unwrap();
            guard.browse_kind = BrowseKind::Weights;
            guard.browse_dir = std::fs::canonicalize(&weights_dir).unwrap();
            guard.browse_entries = listed;
            guard.browse_cursor = 0;
            guard.overlay = OverlayMode::Browse;
            guard.setup.resume_from.clear();
            guard.setup.run_name_custom = false;
            let backend = TestBackend::new(140, 40);
            let mut terminal = Terminal::new(backend).expect("terminal");
            terminal
                .draw(|frame| draw_ui(frame, &guard))
                .expect("draw browse");
            let buf = terminal.backend().buffer();
            let mut saw_gold_folder = false;
            let mut saw_selected_parent = false;
            for y in 0..40 {
                for x in 0..140 {
                    let cell = &buf[(x, y)];
                    if cell.fg == BROWSE_DIR_FG {
                        saw_gold_folder = true;
                    }
                    if cell.bg == BROWSE_DIR_FG && cell.symbol() == "." {
                        saw_selected_parent = true;
                    }
                }
            }
            assert!(saw_gold_folder, "folders stay gold when they are not selected");
            assert!(saw_selected_parent, "selected .. uses the folder colour");
        }
        let nested_idx = state
            .lock()
            .unwrap()
            .browse_entries
            .iter()
            .position(|entry| entry.label == "nested/")
            .expect("nested folder");
        {
            let mut guard = state.lock().unwrap();
            guard.browse_cursor = nested_idx;
        }
        handle_setup_key(&state, KeyCode::Enter);
        {
            let guard = state.lock().unwrap();
            assert_eq!(guard.overlay, OverlayMode::Browse);
            assert!(guard.browse_dir.ends_with("nested"), "{:?}", guard.browse_dir);
            assert_eq!(guard.browse_entries[0].label, "..");
            assert!(guard
                .browse_entries
                .iter()
                .any(|entry| entry.label == "c.pt"));
            assert!(guard.setup.resume_from.is_empty());
        }
        handle_setup_key(&state, KeyCode::Enter);
        {
            let guard = state.lock().unwrap();
            assert!(guard.browse_dir.ends_with("weights"), "{:?}", guard.browse_dir);
            assert!(guard
                .browse_entries
                .iter()
                .any(|entry| entry.label == "b.pt"));
            assert_eq!(guard.overlay, OverlayMode::Browse);
        }
        let file_idx = state
            .lock()
            .unwrap()
            .browse_entries
            .iter()
            .position(|entry| entry.label == "b.pt")
            .expect("b.pt");
        {
            let mut guard = state.lock().unwrap();
            guard.browse_cursor = file_idx;
        }
        handle_setup_key(&state, KeyCode::Enter);
        let picked = state.lock().unwrap().setup.resume_from.clone();
        assert!(picked.ends_with("b.pt"), "{picked}");
        assert!(state.lock().unwrap().setup.run_name.ends_with("-ft"));
        assert_eq!(state.lock().unwrap().overlay, OverlayMode::None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn launcher_clicks_cycle_open_and_target_start() {
        use ratatui::layout::Rect;

        use crate::app::{ArchChoice, FIELD_ARCH, FIELD_DATASET, FIELD_SCALE, FIELD_START};
        use crate::event::apply_launcher_click;
        use crate::ui::launcher::launcher_click_key;

        let area = Rect::new(0, 0, 120, 32);
        let state = AppState::new(CliConfig::default());
        assert!(launcher_click_key(&state, area, 0, 0).is_none());

        let mut scale_left = None;
        let mut scale_right = None;
        let mut scale_enter = None;
        let mut arch_enter = None;
        let mut dataset_enter = None;
        let mut start_enter = None;
        for row in 0..area.height {
            for col in 0..area.width {
                let Some((field, key)) = launcher_click_key(&state, area, col, row) else {
                    continue;
                };
                match (field, key) {
                    (FIELD_SCALE, KeyCode::Left) if scale_left.is_none() => {
                        scale_left = Some((col, row));
                    }
                    (FIELD_SCALE, KeyCode::Right) if scale_right.is_none() => {
                        scale_right = Some((col, row));
                    }
                    (FIELD_SCALE, KeyCode::Enter) if scale_enter.is_none() => {
                        scale_enter = Some((col, row));
                    }
                    (FIELD_ARCH, KeyCode::Enter) if arch_enter.is_none() => {
                        arch_enter = Some((col, row));
                    }
                    (FIELD_DATASET, KeyCode::Enter) if dataset_enter.is_none() => {
                        dataset_enter = Some((col, row));
                    }
                    (FIELD_START, KeyCode::Enter) if start_enter.is_none() => {
                        start_enter = Some((col, row));
                    }
                    _ => {}
                }
            }
        }
        assert!(scale_left.is_some() && scale_right.is_some() && scale_enter.is_some());
        assert!(arch_enter.is_some());
        assert!(dataset_enter.is_some());
        assert!(start_enter.is_some());

        let shared = Arc::new(Mutex::new(state));
        let (col, row) = arch_enter.unwrap();
        apply_launcher_click(&shared, area, col, row);
        {
            let guard = shared.lock().unwrap();
            assert_eq!(guard.setup.arch, ArchChoice::Yolo26);
            assert_eq!(guard.setup.focused_field, FIELD_ARCH);
            assert!(!guard.setup.editing_field);
        }

        let (col, row) = scale_right.unwrap();
        let before = shared.lock().unwrap().setup.model.clone();
        apply_launcher_click(&shared, area, col, row);
        {
            let guard = shared.lock().unwrap();
            assert_eq!(guard.setup.focused_field, FIELD_SCALE);
            assert_ne!(guard.setup.model, before);
            assert!(!guard.setup.editing_field);
        }

        let (col, row) = dataset_enter.unwrap();
        apply_launcher_click(&shared, area, col, row);
        {
            let guard = shared.lock().unwrap();
            assert_eq!(guard.setup.focused_field, FIELD_DATASET);
            assert!(guard.setup.editing_field);
        }
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
        assert!(
            splash_out.contains("Version v0.2.2"),
            "Splash must display Version v0.2.2 under the ASCII art"
        );
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

        // Configure 1-epoch fast XPU run in launcher by cycling scale (medium -> small -> nano) and changing hyperparameters:
        handle_setup_key(&state, KeyCode::Down); // 1: Model Scale (currently yolo11m.pt)
        handle_setup_key(&state, KeyCode::Left); // small (yolo11s.pt)
        handle_setup_key(&state, KeyCode::Left); // nano (yolo11n.pt)
        assert_eq!(state.lock().unwrap().setup.model, "yolo11n.pt");
        {
            let mut guard = state.lock().unwrap();
            guard.setup.dataset = tmp_ds.to_string_lossy().to_string();
            guard.setup.epochs = "1".to_string();
            guard.setup.batch = "2".to_string();
            guard.setup.imgsz = 64;
            guard.setup.imgsz_input = "64".to_string();
            guard.setup.device = "xpu".to_string();
            guard.setup.workers_input = "0".to_string();
            guard.setup.degrees = "90".to_string();
            guard.setup.run_name = "tui-short-verify".to_string();
        }

        // Press F2 in launcher -> transitions to UiMode::Training and spawns run_training + ONNX export
        let should_quit = handle_setup_key(&state, KeyCode::F(2));
        assert!(!should_quit);
        {
            let guard = state.lock().unwrap();
            assert_eq!(guard.mode, UiMode::Training);
            assert!(guard.is_running);
            let req = guard.active_request.as_ref().expect("active_request set");
            assert_eq!(req.model_choice, "yolo11n.pt");
            assert_eq!(req.epochs, 1);
            assert_eq!(req.batch, 2);
            assert_eq!(req.imgsz, 64);
            assert_eq!(req.device, "xpu");
            assert_eq!(req.workers, Some(0));
            assert_eq!(req.degrees, 90.0);
        }

        let deadline = Instant::now() + Duration::from_secs(120);
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
            // Healthy status line and bottom-left yellow version label
            assert!(screen.contains("● Healthy"));
            assert!(screen.contains("Version v0.2.2"));
            let bottom_buf = terminal.backend().buffer();
            assert_eq!(bottom_buf[(1, 33)].symbol(), "V");
            assert_eq!(bottom_buf[(1, 33)].fg, Color::Yellow);
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

    fn multi_xpu_state(device: &str, family: &str) -> AppState {
        let mut cfg = CliConfig::default();
        cfg.interactive = false;
        cfg.dataset = "/tmp".to_string();
        cfg.device = device.to_string();
        cfg.epochs = "60".to_string();
        if family == "YOLO26" {
            cfg.arch = ArchChoice::Yolo26;
            cfg.model = "m".to_string();
        }
        let req = cfg.to_training_request().expect("training request");
        let mut state = AppState::new(cfg);
        state.mode = UiMode::Training;
        state.is_running = true;
        state.multi_xpu_proc_alive = true;
        state.latest_metrics.total_epochs = req.epochs;
        state.active_request = Some(req);
        state
    }

    #[test]
    fn multi_xpu_metrics_file_updates_epoch_map_vram_and_util_gauges() {
        let mut yolo = multi_xpu_state("xpu:0,1", "YOLO");
        assert!(yolo_multi_xpu_request(
            yolo.active_request.as_ref().expect("request")
        ));
        let yolo26 = multi_xpu_state("xpu:0,1", "YOLO26");
        assert_eq!(
            yolo26.active_request.as_ref().expect("request").model_family,
            "YOLO"
        );
        assert!(yolo_multi_xpu_request(
            yolo26.active_request.as_ref().expect("request")
        ));
        let single = multi_xpu_state("xpu:0", "YOLO");
        assert!(!yolo_multi_xpu_request(
            single.active_request.as_ref().expect("request")
        ));
        let mut numeric = multi_xpu_state("0,1", "YOLO");
        assert!(!yolo_multi_xpu_request(
            numeric.active_request.as_ref().expect("request")
        ));
        numeric.active_request.as_mut().expect("request").device = "xpu:0, xpu:1".to_string();
        assert!(yolo_multi_xpu_request(
            numeric.active_request.as_ref().expect("request")
        ));
        let mut rf = multi_xpu_state("xpu:0,1", "YOLO");
        rf.active_request.as_mut().expect("request").model_family = "RF-DETR".to_string();
        assert!(!yolo_multi_xpu_request(
            rf.active_request.as_ref().expect("request")
        ));

        let dir = std::env::temp_dir().join(format!(
            "ls-trainer-metrics-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("metrics.json");
        let missing = dir.join("missing.json");

        yolo.poll_yolo_multi_xpu_metrics_at(&missing, std::time::SystemTime::now());
        assert!(yolo.multi_xpu_gauges_blank);
        assert_eq!(yolo.latest_metrics.epoch, 0);
        assert!(yolo.best_map50.is_none());
        assert!(yolo.best_map50_95.is_none());

        let body = r#"{"epoch":12,"best_map":0.742,"map":0.718,"map50":0.91,"best_map50":0.93,"vram_mb":[2100,1980],"util_pct":[87,91],"gpu_index":[0,1]}"#;
        std::fs::write(&path, body).expect("write metrics");
        let fresh = std::time::SystemTime::now();
        yolo.poll_yolo_multi_xpu_metrics_at(&path, fresh);
        assert!(!yolo.multi_xpu_gauges_blank);
        assert_eq!(yolo.latest_metrics.epoch, 12);
        assert_eq!(yolo.latest_metrics.total_epochs, 60);
        assert!((yolo.best_map50.expect("mAP50") - 0.93).abs() < 1e-9);
        assert!((yolo.best_map50_95.expect("mAP50-95") - 0.742).abs() < 1e-9);
        assert_eq!(yolo.gpus.len(), 2);
        assert!((yolo.gpus[0].vram_used_gb - (2100.0 / 1024.0)).abs() < 1e-9);
        assert!((yolo.gpus[1].vram_used_gb - (1980.0 / 1024.0)).abs() < 1e-9);
        assert!((yolo.gpus[0].util_percent - 87.0).abs() < 1e-9);
        assert!((yolo.gpus[1].util_percent - 91.0).abs() < 1e-9);
        assert!(!yolo.gpus[0].hide_vram && !yolo.gpus[0].hide_util);
        assert_eq!(yolo.train_loss_history.len(), 0);

        let backend = TestBackend::new(160, 40);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_ui(frame, &yolo))
            .expect("draw fresh gauges");
        let screen = buffer_to_string(terminal.backend().buffer());
        assert!(
            screen.contains("Epoch 12 / 60"),
            "epoch gauge missing from:\n{screen}"
        );
        assert!(screen.contains("93.00%"), "mAP50 gauge missing from:\n{screen}");
        assert!(
            screen.contains("74.20%"),
            "mAP50-95 gauge missing from:\n{screen}"
        );
        assert!(
            screen.contains("VRAM 2.05 GiB"),
            "vram gauge missing from:\n{screen}"
        );
        assert!(
            screen.contains("VRAM 1.93 GiB"),
            "second vram gauge missing from:\n{screen}"
        );
        assert!(screen.contains("87.0%"), "util gauge missing from:\n{screen}");
        assert!(screen.contains("91.0%"), "second util gauge missing from:\n{screen}");

        let _ = std::fs::remove_file(&path);
        yolo.poll_yolo_multi_xpu_metrics_at(&path, fresh + std::time::Duration::from_secs(5));
        assert_eq!(yolo.latest_metrics.epoch, 12, "a just-removed file keeps the last reading");

        std::fs::write(&path, "not-json").expect("write garbage");
        yolo.poll_yolo_multi_xpu_metrics_at(&path, fresh);
        assert!(yolo.multi_xpu_gauges_blank);
        assert!(yolo.best_map50.is_none());

        std::fs::write(&path, body).expect("rewrite metrics");
        let again = std::time::SystemTime::now();
        yolo.poll_yolo_multi_xpu_metrics_at(&path, again);
        assert_eq!(yolo.latest_metrics.epoch, 12);
        yolo.poll_yolo_multi_xpu_metrics_at(&path, again + std::time::Duration::from_secs(30));
        assert!(yolo.multi_xpu_gauges_blank);
        assert_eq!(yolo.latest_metrics.epoch, 0);
        terminal
            .draw(|frame| draw_ui(frame, &yolo))
            .expect("draw blank gauges");
        let blank = buffer_to_string(terminal.backend().buffer());
        assert!(
            blank.contains("Epoch -- / 60"),
            "stale file should blank the epoch gauge:\n{blank}"
        );
        assert!(
            blank.contains("VRAM --"),
            "stale file should blank VRAM:\n{blank}"
        );
        assert!(!blank.contains("74.20%"));

        let mut untouched = multi_xpu_state("xpu:0", "YOLO");
        untouched.latest_metrics.epoch = 1;
        std::fs::write(&path, body).expect("rewrite for single-device");
        untouched.poll_yolo_multi_xpu_metrics_at(&path, std::time::SystemTime::now());
        assert_eq!(untouched.latest_metrics.epoch, 1);
        assert!(untouched.gpus.is_empty());
        assert!(!untouched.multi_xpu_gauges_blank);

        rf.latest_metrics.epoch = 2;
        rf.poll_yolo_multi_xpu_metrics_at(&path, std::time::SystemTime::now());
        assert_eq!(rf.latest_metrics.epoch, 2);
        assert!(rf.gpus.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Point `LS_TRAINER_METRICS_SNAPSHOT` at a JSON file captured from a live dual-XPU run.
    #[test]
    fn live_multi_xpu_metrics_snapshot_drives_gauges() {
        let Ok(src) = std::env::var("LS_TRAINER_METRICS_SNAPSHOT") else {
            return;
        };
        let raw = std::fs::read_to_string(&src).expect("read snapshot");
        let value: serde_json::Value = serde_json::from_str(raw.trim()).expect("snapshot json");
        let epoch = value["epoch"].as_u64().expect("epoch") as u32;
        assert!(epoch >= 1, "live snapshot epoch was {epoch}");
        let vram = value["vram_mb"].as_array().expect("vram_mb");
        let util = value["util_pct"].as_array().expect("util_pct");
        assert!(
            vram.len() >= 2,
            "expected both Arc VRAM samples, got {vram:?}"
        );
        assert!(
            util.len() >= 2,
            "expected both Arc util samples, got {util:?}"
        );
        assert!(
            vram.iter().any(|item| item.as_f64().unwrap_or(0.0) > 0.0),
            "expected non-zero VRAM, got {vram:?}"
        );
        let util_nums: Vec<f64> = util.iter().filter_map(|item| item.as_f64()).collect();
        assert!(
            util_nums.len() >= 2,
            "expected numeric util for both devices, got {util:?}"
        );

        let mut state = multi_xpu_state("xpu:0,1", "YOLO");
        let dir = std::env::temp_dir().join(format!(
            "ls-trainer-live-metrics-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("metrics.json");
        std::fs::write(&path, raw.trim()).expect("copy snapshot");
        state.poll_yolo_multi_xpu_metrics_at(&path, std::time::SystemTime::now());
        assert!(!state.multi_xpu_gauges_blank);
        assert_eq!(state.latest_metrics.epoch, epoch.min(60));

        let backend = TestBackend::new(160, 40);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal
            .draw(|frame| draw_ui(frame, &state))
            .expect("draw live snapshot");
        let screen = buffer_to_string(terminal.backend().buffer());
        let epoch_label = format!("Epoch {} / 60", epoch.min(60));
        assert!(
            screen.contains(&epoch_label),
            "epoch gauge missing from:\n{screen}"
        );
        let mb0 = vram[0].as_f64().expect("vram0");
        let vram_label = format!("VRAM {:.2} GiB", mb0 / 1024.0);
        assert!(
            screen.contains(&vram_label),
            "vram gauge {vram_label} missing from:\n{screen}"
        );
        let util_label = format!("{:.1}%", util_nums[0]);
        assert!(
            screen.contains(&util_label),
            "util gauge {util_label} missing from:\n{screen}"
        );
        if let Some(best) = value["best_map"].as_f64() {
            let map_label = format!("{:.2}%", best * 100.0);
            assert!(
                screen.contains(&map_label),
                "best mAP {map_label} missing from:\n{screen}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
