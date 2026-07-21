import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_MODEL_ALIASES = {
    "t5-v1_1-base": "google/t5-v1_1-base",
    "gpt2": "gpt2",
    "SaProt_650M_AF2": "westlake-repl/SaProt_650M_AF2",
}


def _set_env_default(key, value):
    os.environ.setdefault(key, os.path.expanduser(os.path.expandvars(str(value))))


def load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                os.environ.setdefault(key, os.path.expanduser(os.path.expandvars(value)))

    _set_env_default("PROJECT_ROOT", PROJECT_ROOT)
    _set_env_default("DATA_ROOT", PROJECT_ROOT / "data")
    _set_env_default("HF_MODELS_ROOT", os.environ["PROJECT_ROOT"] + "/hf_models")
    _set_env_default("CACHE_ROOT", "~/.cache")
    _set_env_default("OUTPUT_ROOT", os.environ["PROJECT_ROOT"] + "/output")


def env_path(key, *parts):
    load_env()
    return os.path.join(os.environ[key], *parts)


def hf_cache_dir():
    return env_path("HF_MODELS_ROOT")


def resolve_pretrained_path(path_or_name):
    path_or_name = os.path.expanduser(os.path.expandvars(str(path_or_name)))
    if os.path.exists(path_or_name):
        return path_or_name
    model_name = os.path.basename(path_or_name.rstrip(os.sep))
    return HF_MODEL_ALIASES.get(model_name, path_or_name)
