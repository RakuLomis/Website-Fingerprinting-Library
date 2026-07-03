import json
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


DEFAULT_DATASETS = ["cstnet_tls_1.3", "CipherSpectrum"]
DEFAULT_MODEL = "DF"


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_config(config_path=None):
    path = Path(config_path) if config_path else repo_root() / "comparison" / "model_comparison_config.json"
    with path.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    base = repo_root()
    for key in ["raw_root", "generated_root", "results_root", "checkpoints_root"]:
        cfg[key] = str(resolve_repo_relative(cfg[key], base))
    return cfg


def resolve_repo_relative(value, base=None):
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or repo_root()) / path


def display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        try:
            return str(Path(os.path.relpath(path, repo_root())))
        except ValueError:
            return str(path)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, sort_keys=True)


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def run_cmd(cmd, dry_run=False, cwd=None, env=None):
    printable = " ".join(str(part) for part in cmd)
    print(printable)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=cwd, env=env, check=True)
    return completed.returncode


def progress(iterable, **kwargs):
    defaults = {
        "dynamic_ncols": True,
        "file": sys.stderr,
        "leave": True,
        "disable": os.environ.get("WF_PIPELINE_DISABLE_TQDM", "").lower() in {"1", "true", "yes"},
    }
    defaults.update(kwargs)
    return tqdm(iterable, **defaults)


def python_cmd(cfg):
    return [str(part) for part in cfg.get("python_cmd", ["python"])]


def dataset_dir(cfg, dataset):
    return Path(cfg["generated_root"]) / dataset


def result_subdir(cfg, kind, dataset, model_name=None):
    return Path(cfg["results_root"]) / kind / dataset / (model_name or cfg.get("model", DEFAULT_MODEL))


def checkpoint_dir(cfg, dataset, model_name=None):
    return Path(cfg["checkpoints_root"]) / dataset / (model_name or cfg.get("model", DEFAULT_MODEL))


def no_cuda_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def load_df_class():
    df_path = repo_root() / "WFlib" / "models" / "DF.py"
    spec = importlib.util.spec_from_file_location("wflib_df_only", df_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DF


def load_model_class(model_file, class_name):
    model_path = repo_root() / "WFlib" / "models" / model_file
    module_name = f"wflib_{Path(model_file).stem.lower()}_only"
    spec = importlib.util.spec_from_file_location(module_name, model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def instantiate_model(model_cfg, num_classes, num_tabs=1):
    model_cls = load_model_class(model_cfg["file"], model_cfg["class_name"])
    if model_cfg.get("num_tabs_arg"):
        return model_cls(num_classes, num_tabs)
    return model_cls(num_classes)
