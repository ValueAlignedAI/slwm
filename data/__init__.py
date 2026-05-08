"""Data contracts and dependency-light text/code helpers for SLWM-124M."""

from data.contract import (
    MODALITY_IDS,
    REQUIRED_MODALITIES,
    SOURCE_TAGS,
    SignalSample,
    SignalStreamRef,
    validate_sample_contract,
)
from data.text_code import TextCodeDatasetBundle, TextCodeRecord, TokenWindowDataset, build_text_code_lm_datasets
from data.tokenizer import ByteFallbackTokenizer, GPT2BPETokenizer, build_text_tokenizer

__all__ = [
    "MODALITY_IDS",
    "REQUIRED_MODALITIES",
    "SOURCE_TAGS",
    "SignalSample",
    "SignalStreamRef",
    "TextCodeDatasetBundle",
    "TextCodeRecord",
    "TokenWindowDataset",
    "ByteFallbackTokenizer",
    "GPT2BPETokenizer",
    "build_text_code_lm_datasets",
    "build_text_tokenizer",
    "validate_sample_contract",
]
