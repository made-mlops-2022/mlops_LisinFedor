import importlib
import os
import yaml
import logging
import inspect
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Callable
from ml_project.project_paths import MODEL_CONFIGS_PATH


logger = logging.getLogger(__name__)


@dataclass()
class ModelParams:
    last_model: Optional[str]
    module: str = field(default="sklearn")
    submodule: Optional[str] = field(default="ensemble")
    model_name: str = field(default="RandomForestClassifier")
    kwargs_path: Optional[str] = field(default="")


def create_clear_model(params: ModelParams) -> Any:
    """Create model from pretrained import from lib.

    Args:
        params (ModelParams): parameters to find model.

    Returns:
        Any: class of model with `fit` and `predict` methods.
    """
    if params.submodule is None:
        model_path = params.module
    else:
        model_path = ".".join([params.module, params.submodule])
    
    module = importlib.import_module(model_path)
    return getattr(module, params.model_name)


def init_model(model_class: Any, params: ModelParams) -> Any:
    if params.kwargs_path:
        with open(params.kwargs_path, "r") as kwargs_file:
            kwargs = yaml.safe_load(kwargs_file)
        return model_class(**kwargs)

    logger.warning("kwargs_path is null. Use default params.")
    logger.warning("See default params in model_configs.")
    def_kwargs = get_default_kwargs(model_class, params.model_name)
    return model_class(**def_kwargs)


def save_def_kwargs(kwargs: Dict[str, Any], file_path: Path):
    logger.warning("Create file with default kwars.")
    logger.warning(f"File created: {file_path}")
    with open(file_path, "w") as kw_file:
        yaml.safe_dump(kwargs, kw_file)


def get_default_kwargs(func: Callable, model_name: str):
    """Get default model args."""
    file_name = "{model_name}_kwargs.yml".format(model_name=model_name)
    file_path = MODEL_CONFIGS_PATH / file_name

    if os.path.exists(file_path):
        with open(file_path, "r") as kwargs_file:
            return yaml.safe_load(kwargs_file)

    logger.warning(
        f"Can't find default model kwargs file {file_name} in configs."
    )
    signature = inspect.signature(func)
    def_kwargs = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    save_def_kwargs(def_kwargs, file_path)

    return def_kwargs

