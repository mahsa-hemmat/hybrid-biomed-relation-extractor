import argparse
import logging
import subprocess
import os
import sys

# --- CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("main")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "parser":              os.path.join(BASE_DIR, "data_parser.py"),
    "preprocess":          os.path.join(BASE_DIR, "preprocess.py"),
    "train":               os.path.join(BASE_DIR, "train_models.py"),
    "evaluate":            os.path.join(BASE_DIR, "evaluate_models.py"),
    "predict":             os.path.join(BASE_DIR, "run_inference.py"),
    "ensemble":            os.path.join(BASE_DIR, "run_ensemble_logic.py"),
    "validate_hard_cases": os.path.join(BASE_DIR, "run_biomistral_judge.py"),
    "results_report":      os.path.join(BASE_DIR, "csvToJson.py"),
}

def run_script(script_path: str, extra_args: list[str] = ()) -> None:
    """Execute *script_path* as a subprocess and propagate exit codes."""
    if not os.path.exists(script_path):
        log.error("Script not found: %s", script_path)
        sys.exit(1)

    cmd = [sys.executable, script_path] + list(extra_args)
    log.info("Starting: %s", " ".join(cmd))
    log.info("-" * 60)

    try:
        subprocess.check_call(cmd)
        log.info("-" * 60)
        log.info("Finished: %s", script_path)
    except subprocess.CalledProcessError as exc:
        log.error("Module failed (exit code %d): %s", exc.returncode, script_path)
        sys.exit(exc.returncode)
        

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Biomedical Relation Extraction — pipeline orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True, help="Pipeline stage")

    sub.add_parser("parser",  help="Parse RegulaTome/BioRED and build training data")

    p_prep = sub.add_parser("preprocess", help="Add entity markers to a raw CSV")
    p_prep.add_argument("--file", default="data/results_filtered.csv", help="Input CSV path")

    p_train = sub.add_parser("train", help="Fine-tune BERT models on processed data")
    p_train.add_argument("--epochs", default="6",    help="Training epochs (default: 6)")
    p_train.add_argument("--lr",     default="2e-5", help="Learning rate (default: 2e-5)")

    sub.add_parser("evaluate",            help="Evaluate trained models on the test set")

    p_pred = sub.add_parser("predict",    help="Run inference on a new CSV file")
    p_pred.add_argument("--input", default="data/processed_data.csv", help="Input CSV path")

    sub.add_parser("ensemble",            help="Apply majority-vote ensemble logic")
    sub.add_parser("validate_hard_cases", help="Adjudicate hard cases with BioMistral-7B")
    sub.add_parser("results_report",      help="Generate final JSON report")

    args = parser.parse_args()

    dispatch = {
        "parser":              (SCRIPTS["parser"],              []),
        "preprocess":          (SCRIPTS["preprocess"],          ["--file", args.file] if args.mode == "preprocess" else []),
        "train":               (SCRIPTS["train"],               ["--epochs", args.epochs, "--lr", args.lr] if args.mode == "train" else []),
        "evaluate":            (SCRIPTS["evaluate"],            []),
        "predict":             (SCRIPTS["predict"],             ["--input", args.input] if args.mode == "predict" else []),
        "ensemble":            (SCRIPTS["ensemble"],            []),
        "validate_hard_cases": (SCRIPTS["validate_hard_cases"], []),
        "results_report":      (SCRIPTS["results_report"],      []),
    }

    script, extra = dispatch[args.mode]
    run_script(script, extra)


if __name__ == "__main__":
    main()

"""
DATASET ACKNOWLEDGEMENT:
This software utilizes the following datasets for training:
1. RegulaTome: Nastou et al. (2024) - https://doi.org/10.5281/zenodo.10808330
2. BioRED: Luo et al. (2022) - https://doi.org/10.1093/bib/bbac282
"""
