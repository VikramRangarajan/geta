import importlib.util
import logging
import os
import sys


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bootstrap_paths():
    root = repo_root()
    for p in (root, os.path.join(root, "test_scripts"), os.path.join(root, "tutorials")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def resolve_output_dir(explicit=None, label="run"):
    if explicit:
        out = explicit
    else:
        out = os.environ.get("GETA_OUTPUT_DIR", os.path.join(repo_root(), "outputs", label))
    os.makedirs(out, exist_ok=True)
    return out


def resolve_data_dir(explicit=None):
    if explicit:
        return explicit
    return os.environ.get("GETA_DATA_DIR", os.path.join(repo_root(), "data"))


def add_common_args(parser):
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    return parser


def load_check_accuracy():
    try:
        from utils.utils import check_accuracy

        return check_accuracy
    except Exception:
        pass
    path = os.path.join(repo_root(), "tutorials", "utils", "utils.py")
    spec = importlib.util.spec_from_file_location("geta_tutorial_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.check_accuracy


def create_exp_dir(config, outputs="outputs", exp_name="exp"):
    base = getattr(config, "output_dir", None) or os.environ.get("GETA_OUTPUT_DIR")
    if not base:
        base = os.path.join(repo_root(), outputs, exp_name)
    os.makedirs(base, exist_ok=True)
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(base, f"{exp_name}.log"))
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(sh)
    return logger
