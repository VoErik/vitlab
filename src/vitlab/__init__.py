"""vitlab -- a reproducible ViT foundation.

    vitlab.config      DATA_ROOT lives here. Edit it once.
    vitlab.registry    the model zoo, pinned to commit SHAs
    vitlab.datasets    the dataset registry + loading (works without any training)
    vitlab.backbone    load a pinned HF vision tower, uniform forward pass
    vitlab.activations named activation sites (feeds overcomplete / nnsight)
    vitlab.heads       task heads
    vitlab.model       MultiTaskViT + save/load
    vitlab.batching    task-homogeneous multi-task batching
    vitlab.train       the training loop
    vitlab.lora        LoRA via PEFT
"""

from .activations import ActivationReader, extract_activations, ActivationStore
from .arch import ARCHS, ArchSpec, get_arch
from .backbone import BackboneOutput, VisionBackbone, load_image_processor
from .batching import MultiTaskLoader, TaskBatch, collate, make_loader
from .config import DATA_ROOT, get_data_root, set_data_root
from .heads import TaskHead, TaskSpec
from .lora import LoraConfig, apply_lora, trainable_parameter_summary
from .model import ModelConfig, MultiTaskViT, load_model, save_model
from .registry import REGISTRY, ModelSpec, get_spec, list_models
from .utils.runtime import seed_everything, load_dotenv
from .datasets import get_splits

__all__ = [
    # utilities
    "seed_everything", "load_dotenv",
    # data
    "DATA_ROOT", "set_data_root", "get_data_root", "get_splits",
    "MultiTaskLoader", "TaskBatch", "make_loader", "collate",
    # models
    "REGISTRY", "ModelSpec", "get_spec", "list_models",
    "VisionBackbone", "BackboneOutput", "load_image_processor",
    "ARCHS", "ArchSpec", "get_arch",
    "ActivationReader", "ActivationStore", "extract_activations",
    "TaskSpec", "TaskHead",
    "ModelConfig", "MultiTaskViT", "save_model", "load_model",
    "LoraConfig", "apply_lora", "trainable_parameter_summary",
]
