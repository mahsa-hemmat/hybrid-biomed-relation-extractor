import os

#  Directory Layout 
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
RAW_DIR    = os.path.join(DATA_DIR, "raw")
PROC_DIR   = os.path.join(DATA_DIR, "processed")

RESULTS_DIR     = os.path.join(BASE_DIR, "results")
MODELS_DIR      = os.path.join(RESULTS_DIR, "models")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")
ANALYSIS_DIR    = os.path.join(RESULTS_DIR, "analysis")

#  Raw Data Sources 
REGULATOME_DIR = os.path.join(RAW_DIR, "RegulaTome-corpus")
BIORED_DIR     = os.path.join(RAW_DIR, "BioRED", "BioRED")
BIORED_FILES   = {
    "train": "Train.PubTator",
    "dev":   "Dev.PubTator",
    "test":  "Test.PubTator",
}

REGULATOME_URL = "https://zenodo.org/records/10808330/files/RegulaTome-corpus.tar.gz?download=1"
BIORED_URL     = "https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip"

#  Label Mapping 
UNIFIED_MAPPING = {
    "Positive_regulation":  "Positive_regulation",
    "Positive_Correlation": "Positive_regulation",
    "Negative_regulation":  "Negative_regulation",
    "Negative_Correlation": "Negative_regulation",
    "Out-of-scope":         "NO_RELATION",
    "Comparison":           "NO_RELATION",
    "Cotreatment":          "NO_RELATION",
}

#  Data Balancing (Training Only) 
PRUNE_THRESHOLD  = 20    # Drop classes with fewer than this many training samples
NEGATIVE_RATIO   = 4.0   # Negative-to-positive sample ratio (1:N)

#  Model Registry 
PRETRAINED_MODELS = {
    "BioBERT":    "dmis-lab/biobert-base-cased-v1.2",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "BioLinkBERT":"michiyasunaga/BioLinkBERT-base",
}

TRAINED_MODELS = {
    name: os.path.join(MODELS_DIR, name, "final_model")
    for name in PRETRAINED_MODELS
}

#  Training Hyperparameters 
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE  = 32
GRAD_ACCUM_STEPS = 2
WEIGHT_DECAY     = 0.02
WARMUP_RATIO     = 0.15
LR_SCHEDULER     = "linear"
MAX_SEQ_LENGTH   = 256
EARLY_STOP_PATIENCE = 2
SAVE_TOTAL_LIMIT = 2
RANDOM_SEED      = 42

#  Inference / Ensemble 
INFERENCE_BATCH_SIZE = 32
CONF_THRESHOLD       = 0.80   # Minimum confidence for a model vote to count

#  LLM Judge 
JUDGE_MODEL_ID = "BioMistral/BioMistral-7B"

#  Pipeline File Paths 
HARD_CASES_CSV           = os.path.join(ANALYSIS_DIR, "hard_cases_for_llm_judge.csv")
ADJUDICATED_CSV          = os.path.join(ANALYSIS_DIR, "final_adjudicated_relations.csv")
ENSEMBLE_REPORT_CSV      = os.path.join(ANALYSIS_DIR, "final_ensemble_report.csv")
FULL_COMPARISON_CSV      = os.path.join(ANALYSIS_DIR, "full_model_comparison.csv")
FINAL_REPORT_CSV         = os.path.join(ANALYSIS_DIR, "final_report.csv")
FINAL_JSON               = os.path.join(ANALYSIS_DIR, "ppi_final.json")
PREP_LOG                 = os.path.join(PROC_DIR,     "prep_report.txt")

#  Shared columns used across ensemble files 
JOIN_KEYS = ["PubMedID", "Entity_1", "Entity_2", "Sentence"]
CORE_COLS = JOIN_KEYS
MODEL_COLS = [f"{m}_Relation" for m in PRETRAINED_MODELS]
