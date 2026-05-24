"""CLI: ``python -m orchestration <command>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_smoke(_args: argparse.Namespace) -> None:
    from .integration_smoke import main as smoke_main

    smoke_main()


def _cmd_validate(args: argparse.Namespace) -> None:
    from .config_validate import validate_pipeline_config

    err, warns = validate_pipeline_config(Path(args.config))
    for w in warns:
        print("WARN:", w)
    if err:
        for e in err:
            print("ERROR:", e)
        sys.exit(1)
    print("OK:", args.config)


def _cmd_from_parquet(args: argparse.Namespace) -> None:
    from .glue import run_from_parquet_report, run_wfo_backtest_from_parquet
    from .real_data_benchmark import write_benchmark_artifacts

    config_path = Path(args.config)
    if args.json_out or args.csv_out:
        report = run_from_parquet_report(
            args.path,
            config_path,
            raw_ohlcv=args.raw_ohlcv,
            light_only=not args.full_models,
            dl_epochs=args.dl_epochs,
        )
        folds = report["fold_metrics"]
    else:
        folds = run_wfo_backtest_from_parquet(
            args.path,
            config_path,
            raw_ohlcv=args.raw_ohlcv,
            light_only=not args.full_models,
            dl_epochs=args.dl_epochs,
        )
        report = None

    print(folds.to_string(index=False))

    if report is not None:
        print("\n--- Summary ---")
        for k, v in report["summary"].items():
            print(f"  {k}: {v}")
        for level in ("acceptance", "target"):
            ev = report["criteria"][level]
            status = "PASS" if ev["passed"] else "FAIL"
            print(f"\n--- {level.upper()} ({status}) ---")
            for chk in ev["checks"]:
                mark = "OK" if chk["passed"] else "X"
                print(f"  [{mark}] {chk['name']}: {chk['detail']}")
        if args.json_out:
            write_benchmark_artifacts(report, json_out=Path(args.json_out))
            print(f"\nWrote JSON: {args.json_out}")
        if args.csv_out:
            csv_path = Path(args.csv_out)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            folds.to_csv(csv_path, index=False)
            print(f"Wrote CSV: {csv_path}")


def _cmd_report_real(args: argparse.Namespace) -> None:
    """OHLCV parquet → FeatureEngine → WFO backtest + criteria (canonical 4-model by default)."""
    from backtesting.results_journal import BacktestResultsJournal

    from .real_data_benchmark import (
        default_real_parquet,
        run_benchmark_report,
        write_benchmark_artifacts,
    )

    from orchestration.benchmark_runner import load_tuning_best_params

    symbol = getattr(args, "symbol", None)
    timeframe = getattr(args, "timeframe", None) or "1h"
    if args.parquet:
        pq = Path(args.parquet)
    elif symbol:
        from orchestration.symbols import paths_for

        pq = paths_for(symbol, timeframe).parquet
        if not pq.is_file():
            raise FileNotFoundError(
                f"OHLCV not found for {symbol} {timeframe}: {pq}. "
                "Run prepare-symbol with --download or place parquet under data/ohlcv/."
            )
    else:
        pq = default_real_parquet()
    use_best = bool(getattr(args, "use_tuning_best", True))
    config_path = args.config
    tuning: dict = {}
    if use_best:
        tuning = load_tuning_best_params(args.config, symbol=symbol, timeframe=timeframe) or {}
        if tuning.get("config_path"):
            config_path = tuning["config_path"]
    if args.max_rows == 0:
        max_rows = int(tuning["max_rows"]) if tuning.get("max_rows") else None
    else:
        max_rows = args.max_rows
    cfg = None
    if not use_best:
        from orchestration.benchmark_runner import orchestrator_config_from_yaml

        cfg = orchestrator_config_from_yaml(config_path)
    dl_epochs = int(getattr(args, "dl_epochs", 5))
    if tuning:
        dl_epochs = int(tuning.get("dl_epochs", dl_epochs))
    report = run_benchmark_report(
        pq,
        max_rows=max_rows,
        config_path=config_path,
        orchestrator_config=cfg,
        use_tuning_best=use_best,
        dl_epochs=dl_epochs,
        symbol=symbol,
        timeframe=timeframe,
        label="report-real",
        use_feature_cache=getattr(args, "use_feature_cache", False),
    )
    folds = report["fold_metrics"]
    print(folds.to_string(index=False))
    print("\n--- Summary ---")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    for level in ("acceptance", "target"):
        ev = report["criteria"][level]
        status = "PASS" if ev["passed"] else "FAIL"
        print(f"\n--- {level.upper()} ({status}) ---")
        for chk in ev["checks"]:
            mark = "OK" if chk["passed"] else "X"
            print(f"  [{mark}] {chk['name']}: {chk['detail']}")
    journal = BacktestResultsJournal()
    if args.json_out:
        write_benchmark_artifacts(
            report,
            json_out=Path(args.json_out),
            journal=journal,
            journal_label="report-real",
        )
    else:
        rid = write_benchmark_artifacts(report, journal=journal, journal_label="report-real")
        print(f"\nJournal run_id: {rid}")


def _resolve_symbol_timeframe(args: argparse.Namespace, *, command: str) -> tuple[str, str]:
    """Positional SYMBOL [TIMEFRAME] or ``--symbol`` / ``--timeframe``."""
    symbol = getattr(args, "symbol_flag", None) or getattr(args, "symbol", None)
    if not symbol:
        raise SystemExit(
            f"{command}: provide SYMBOL (positional) or --symbol, e.g.\n"
            f"  python -m orchestration {command} BTC/USDT 1h --download\n"
            f"  python -m orchestration {command} --symbol BTC/USDT --timeframe 1h --download"
        )
    timeframe = getattr(args, "timeframe_flag", None) or getattr(args, "timeframe", None) or "1h"
    return symbol, timeframe


def _cmd_prepare_symbol(args: argparse.Namespace) -> None:
    import json as _json

    from .symbol_pipeline import prepare_symbol

    symbol, timeframe = _resolve_symbol_timeframe(args, command="prepare-symbol")

    def _print(step: str, payload: dict) -> None:
        if not args.quiet:
            keys = ", ".join(f"{k}={payload[k]}" for k in payload if k not in {"summary", "best_summary"})
            print(f"[{step}] {keys}")

    out = prepare_symbol(
        symbol,
        timeframe,
        download=args.download,
        start_date=args.start_date,
        end_date=args.end_date,
        holdout_fraction=args.holdout_fraction,
        max_trials=args.max_trials,
        do_train_final=not args.skip_final,
        config_path=args.config,
        allow_two_model_override=getattr(args, "allow_two_model", False),
        on_step=_print,
    )
    print(_json.dumps({
        "symbol": out["symbol"],
        "timeframe": out["timeframe"],
        "manifest_path": str(Path("artifacts") / out["manifest"]["slug"] / "manifest.json"),
        "ready_for_paper": out["manifest"]["ready_for_paper"],
        "ready_for_live": out["manifest"]["ready_for_live"],
        "holdout_acceptance_passed": out["manifest"]["holdout_acceptance_passed"],
        "holdout_target_passed": out["manifest"]["holdout_target_passed"],
        "bundle_run_id": out["manifest"]["bundle_run_id"],
        "task_id": out["task_id"],
    }, indent=2, ensure_ascii=False))


def _cmd_explain(args: argparse.Namespace) -> None:
    import json as _json

    from .introspect import explain_symbol

    out = explain_symbol(args.symbol, args.timeframe, window=args.window)
    print(_json.dumps(out, indent=2, ensure_ascii=False, default=str))


def _cmd_train_final_symbol(args: argparse.Namespace) -> None:
    import json as _json

    from .symbol_pipeline import (
        build_features,
        read_symbol_manifest,
        train_final_for_symbol,
        write_symbol_manifest,
        SymbolManifest,
        utc_now_iso,
        new_run_id,
    )
    from .symbol_pipeline import _span_of
    from .symbols import paths_for, tuning_best_for

    sp = paths_for(args.symbol, args.timeframe)
    if not sp.parquet.is_file():
        print(f"parquet not found: {sp.parquet}")
        sys.exit(2)
    feat = build_features(sp.parquet)
    rid = new_run_id()
    bundle_dir = train_final_for_symbol(sp, feat, run_id=rid)
    best_params = tuning_best_for(sp.symbol, sp.timeframe)

    existing = read_symbol_manifest(sp) or {}
    manifest = SymbolManifest(
        symbol=sp.symbol,
        timeframe=sp.timeframe,
        slug=sp.slug,
        created_at=utc_now_iso(),
        parquet=str(sp.parquet),
        train_span=existing.get("train_span") or _span_of(feat),
        holdout_span=existing.get("holdout_span") or {"start": None, "end": None, "rows": 0},
        full_span=_span_of(feat),
        tuning_best_params=best_params or existing.get("tuning_best_params") or {},
        tune_run_ids=existing.get("tune_run_ids") or [],
        holdout_run_id=existing.get("holdout_run_id"),
        holdout_summary=existing.get("holdout_summary") or {},
        holdout_acceptance_passed=bool(existing.get("holdout_acceptance_passed")),
        holdout_target_passed=bool(existing.get("holdout_target_passed")),
        artifact_bundle=str(bundle_dir),
        bundle_run_id=rid,
        ready_for_paper=bool(existing.get("holdout_acceptance_passed")),
        ready_for_live=False,
        notes=("forced" if args.force else "") + (existing.get("notes") or ""),
    )
    write_symbol_manifest(sp, manifest)
    print(_json.dumps({
        "symbol": sp.symbol,
        "timeframe": sp.timeframe,
        "bundle": str(bundle_dir),
        "ready_for_paper": manifest.ready_for_paper,
        "ready_for_live": manifest.ready_for_live,
    }, indent=2, ensure_ascii=False))


def _cmd_regime_history(args: argparse.Namespace) -> None:
    import json as _json

    from .introspect import regime_history

    rows = regime_history(
        args.symbol,
        args.timeframe,
        start=args.start,
        end=args.end,
        step=args.step,
    )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            _json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} rows -> {args.json_out}")
    else:
        for r in rows[-20:]:
            print(f"{r['t']}  {r['regime']:6}  close={r.get('close')}")
        print(f"... total {len(rows)} rows")


def _cmd_list_symbols(_args: argparse.Namespace) -> None:
    import json as _json

    from .symbols import list_known_symbols

    print(_json.dumps(list_known_symbols(), indent=2, ensure_ascii=False))


def _cmd_manifest_show(args: argparse.Namespace) -> None:
    import json as _json

    from .symbol_pipeline import read_symbol_manifest
    from .symbols import paths_for

    sp = paths_for(args.symbol, args.timeframe)
    man = read_symbol_manifest(sp)
    if man is None:
        print(f"No manifest for {sp.slug}")
        sys.exit(2)
    print(_json.dumps(man, indent=2, ensure_ascii=False))


def _cmd_build_features(args: argparse.Namespace) -> None:
    import json as _json

    from .feature_store import build_features, read_manifest

    df = build_features(
        args.symbol,
        args.timeframe,
        config_path=args.config,
        force=args.force,
    )
    man = read_manifest(args.symbol, args.timeframe)
    print(
        _json.dumps(
            {
                "rows": len(df),
                "columns": len(df.columns),
                "manifest": man,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def _cmd_tune_thesis(args: argparse.Namespace) -> None:
    import json as _json

    from .dl_training import configure_tf_runtime
    from .thesis_tuning import run_multilevel

    configure_tf_runtime()

    use_cache = not getattr(args, "no_feature_cache", False)
    out = run_multilevel(
        args.symbol,
        args.timeframe,
        phase=args.phase,
        config_path=args.config,
        journal_root=args.journal_root,
        use_feature_cache=use_cache,
        tuning_yaml=getattr(args, "tuning_yaml", None),
        tuning_profile=getattr(args, "profile", "default"),
    )
    print(_json.dumps(out, indent=2, ensure_ascii=False, default=str))
    confirm = out.get("confirm") or {}
    if args.phase in ("confirm", "all") and not confirm.get("acceptance_passed"):
        raise SystemExit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Orchestration pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="Synthetic WFO + Backtester (no external data)").set_defaults(
        func=_cmd_smoke
    )

    pv = sub.add_parser("validate-config", help="Validate config.yaml sections")
    pv.add_argument("--config", default="config.yaml", help="Path to YAML")
    pv.set_defaults(func=_cmd_validate)

    pp = sub.add_parser(
        "from-parquet",
        help="Feature parquet → models → WFO backtest (use --full-models for GRU/CNN)",
    )
    pp.add_argument("path", help="Path to .parquet")
    pp.add_argument("--config", default="config.yaml")
    pp.add_argument(
        "--raw-ohlcv",
        action="store_true",
        help="Run FeatureManager transform (expects raw OHLCV input)",
    )
    pp.add_argument(
        "--full-models",
        action="store_true",
        help="Use all model_keys from config (requires TensorFlow for gru/cnn)",
    )
    pp.add_argument("--dl-epochs", type=int, default=3)
    pp.add_argument(
        "--json-out",
        default=None,
        help="Save full report (all folds, summary, acceptance/target) to JSON",
    )
    pp.add_argument(
        "--csv-out",
        default=None,
        help="Save per-fold metrics table to CSV",
    )
    pp.set_defaults(func=_cmd_from_parquet)

    pr = sub.add_parser(
        "report-real",
        help="Train LightGBM+XGBoost on OHLCV parquet + WFO metrics + criteria",
    )
    pr.add_argument(
        "--parquet",
        default=None,
        help="OHLCV parquet; default: data/ohlcv/BTC-USDT_1h.parquet if present",
    )
    pr.add_argument(
        "--config",
        default="config.yaml",
        help="YAML with backtesting.acceptance / backtesting.target",
    )
    pr.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Tail-N feature rows; 0 = orchestration_tuning_best.max_rows or all rows",
    )
    pr.add_argument("--train-window", type=int, default=900)
    pr.add_argument("--test-window", type=int, default=180)
    pr.add_argument("--wfo-step", type=int, default=360)
    pr.add_argument(
        "--json-out",
        default=None,
        help="Write folds + criteria JSON (e.g. docs/e2e_last_metrics.json)",
    )
    pr.add_argument(
        "--use-tuning-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use orchestration_tuning_best from symbol YAML + reference (default: on)",
    )
    pr.add_argument("--symbol", default=None, help="e.g. BTC/USDT — load config/symbols/<slug>.yaml")
    pr.add_argument("--timeframe", default="1h")
    pr.add_argument("--dl-epochs", type=int, default=3, help="GRU/CNN epochs per WFO fold (thesis: 2–3)")
    pr.add_argument(
        "--full-models",
        action="store_true",
        help="Force all model_keys from canonical config (requires TensorFlow for gru/cnn)",
    )
    pr.add_argument(
        "--use-feature-cache",
        action="store_true",
        help="Load data/features/<slug>.parquet instead of inline FeatureEngine",
    )
    pr.set_defaults(func=_cmd_report_real)

    pbf = sub.add_parser(
        "build-features",
        help="Build canonical feature parquet + manifest for a symbol",
    )
    pbf.add_argument("--symbol", required=True, help="e.g. BTC/USDT")
    pbf.add_argument("--timeframe", default="1h")
    pbf.add_argument(
        "--config",
        default="config/profiles/canonical_4model.yaml",
    )
    pbf.add_argument("--force", action="store_true", help="Rebuild even if cache valid")
    pbf.set_defaults(func=_cmd_build_features)

    ptt = sub.add_parser(
        "tune-thesis",
        help="Multilevel thesis tune: fast → refine → confirm",
    )
    ptt.add_argument("--symbol", required=True)
    ptt.add_argument("--timeframe", default="1h")
    ptt.add_argument(
        "--phase",
        choices=("fast", "refine", "confirm", "all"),
        default="all",
    )
    ptt.add_argument(
        "--config",
        default="config/profiles/canonical_4model.yaml",
    )
    ptt.add_argument("--journal-root", default="docs/backtest_journal")
    ptt.add_argument(
        "--no-feature-cache",
        action="store_true",
        help="Rebuild features inline from OHLCV each run",
    )
    ptt.add_argument(
        "--tuning-yaml",
        default=None,
        help="Override thesis_tuning levels (default: config/profiles/thesis_tuning.yaml)",
    )
    ptt.add_argument(
        "--profile",
        choices=("default", "turbo"),
        default="default",
        help="turbo = config/profiles/thesis_tuning_turbo.yaml (fewer folds/trials, tabular-first)",
    )
    ptt.set_defaults(func=_cmd_tune_thesis)

    pt = sub.add_parser(
        "tune-until",
        help="Grid search on real OHLCV until acceptance (and optionally target) criteria pass",
    )
    pt.add_argument("--parquet", default=None)
    pt.add_argument("--config", default="config.yaml")
    pt.add_argument("--max-rows", type=int, default=8000)
    pt.add_argument("--max-trials", type=int, default=80)
    pt.add_argument(
        "--require-target",
        action="store_true",
        help="Do not stop until target (b) criteria pass, not only acceptance (a)",
    )
    pt.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best orchestration params into config.yaml",
    )
    pt.add_argument(
        "--journal-root",
        default="docs/backtest_journal",
    )
    pt.add_argument(
        "--mode",
        choices=("grid", "refine"),
        default="refine",
        help="grid=wide search; refine=local search around orchestration_tuning_best",
    )
    pt.set_defaults(func=_cmd_tune_until)

    # ---- prepare-symbol (download + features + tune + holdout + final fit) ----
    pps = sub.add_parser(
        "prepare-symbol",
        help="End-to-end per-symbol: download → features → tune → holdout → train-final → manifest",
    )
    pps.add_argument(
        "symbol",
        nargs="?",
        metavar="SYMBOL",
        help="Trading pair, e.g. BTC/USDT (or use --symbol)",
    )
    pps.add_argument(
        "timeframe",
        nargs="?",
        default="1h",
        metavar="TIMEFRAME",
        help="Candle timeframe (default: 1h)",
    )
    pps.add_argument("--symbol", dest="symbol_flag", default=None, help="Trading pair (alternative to positional)")
    pps.add_argument(
        "--timeframe",
        dest="timeframe_flag",
        default=None,
        help="Timeframe override (default: 1h)",
    )
    pps.add_argument("--download", action="store_true", help="Force re-download OHLCV")
    pps.add_argument(
        "--config",
        default="config/profiles/canonical_4model.yaml",
        help="Canonical profile (dates, MTF, feature_engineering, orchestration)",
    )
    pps.add_argument("--start-date", default="2022-01-01 00:00:00")
    pps.add_argument("--end-date", default=None)
    pps.add_argument(
        "--allow-two-model",
        action="store_true",
        help="Allow symbol YAML to persist 2-model model_keys from tuning",
    )
    pps.add_argument("--holdout-fraction", type=float, default=0.2)
    pps.add_argument("--max-trials", type=int, default=30)
    pps.add_argument("--skip-final", action="store_true", help="Do not run final fit")
    pps.add_argument("--quiet", action="store_true")
    pps.set_defaults(func=_cmd_prepare_symbol)

    # ---- train-final-symbol (manual final fit) ----
    ptf = sub.add_parser(
        "train-final-symbol",
        help="Force final fit on full history for a symbol (skips holdout gate).",
    )
    ptf.add_argument("--symbol", required=True)
    ptf.add_argument("--timeframe", default="1h")
    ptf.add_argument("--force", action="store_true", help="Mark manifest with notes=forced")
    ptf.set_defaults(func=_cmd_train_final_symbol)

    # ---- explain ----
    pex = sub.add_parser(
        "explain",
        help="Show current decision plan for a symbol (regime, weights, model probs, decision)",
    )
    pex.add_argument("--symbol", required=True)
    pex.add_argument("--timeframe", default="1h")
    pex.add_argument("--window", type=int, default=256)
    pex.set_defaults(func=_cmd_explain)

    # ---- regime-history ----
    prh = sub.add_parser(
        "regime-history",
        help="Per-bar regime label for a date range (for GUI overlay)",
    )
    prh.add_argument("--symbol", required=True)
    prh.add_argument("--timeframe", default="1h")
    prh.add_argument("--start", default=None)
    prh.add_argument("--end", default=None)
    prh.add_argument("--step", type=int, default=1)
    prh.add_argument("--json-out", default=None)
    prh.set_defaults(func=_cmd_regime_history)

    # ---- list-symbols / manifest-show ----
    sub.add_parser(
        "list-symbols",
        help="List symbols with parquet/config/artifact status (JSON for GUI)",
    ).set_defaults(func=_cmd_list_symbols)

    pms = sub.add_parser(
        "manifest-show",
        help="Show artifacts/<slug>/manifest.json (per-symbol summary)",
    )
    pms.add_argument("--symbol", required=True)
    pms.add_argument("--timeframe", default="1h")
    pms.set_defaults(func=_cmd_manifest_show)

    args = p.parse_args()
    args.func(args)


def _cmd_tune_until(args: argparse.Namespace) -> None:
    from .tuning_loop import apply_best_params_to_config_yaml, tune_until_criteria

    out = tune_until_criteria(
        Path(args.parquet) if args.parquet else None,
        max_rows=None if args.max_rows == 0 else args.max_rows,
        max_trials=args.max_trials,
        stop_on_target=args.require_target,
        mode=args.mode,
        journal_root=args.journal_root,
        config_path=args.config,
    )
    print("Trials:", out["trials_run"])
    print("Acceptance achieved:", out["acceptance_achieved"])
    print("Target achieved:", out["target_achieved"])
    print("Journal:", out["journal_root"])
    if out.get("best_params"):
        print("Best params:", out["best_params"])
    if out.get("best_report"):
        s = out["best_report"]["summary"]
        print(
            "Best summary: sharpe={:.3f} pf={:.3f} wfe={:.3f} ret={:.2f}%".format(
                s.get("mean_sharpe", 0),
                s.get("mean_profit_factor", 0),
                s.get("mean_wfe", 0),
                s.get("mean_total_return_pct", 0),
            )
        )
    if args.apply_best and out.get("best_params"):
        apply_best_params_to_config_yaml(out["best_params"], Path(args.config))
        print("Updated", args.config)
    if not out["acceptance_achieved"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
