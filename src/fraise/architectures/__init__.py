"""Architecture contracts and the registry used by training and inference."""

from .contracts import ArchitectureSpec as ArchitectureSpec
from .contracts import ModelConfigProtocol as ModelConfigProtocol
from .contracts import ModelProtocol as ModelProtocol
from .contracts import OptimizerLayout as OptimizerLayout
from .registry import build_model as build_model
from .registry import get_architecture as get_architecture
from .registry import list_architectures as list_architectures
from .registry import model_config_from_dict as model_config_from_dict
from .registry import register_architecture as register_architecture
