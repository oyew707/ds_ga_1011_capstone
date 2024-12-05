"""
-------------------------------------------------------
Dataset and activation extraction classes for LLMs
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "10/28/24"
-------------------------------------------------------
"""

# Imports
import warnings
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Any
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import prepare_model_for_kbit_training
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from transformers.models.auto.tokenization_auto import PreTrainedTokenizerFast
from src.logger import getlogger

# Constants
warnings.filterwarnings('ignore')
log = getlogger(__name__, 'info')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class DataConfig:
    """
    -------------------------------------------------------
    Configuration for LLM loading and activation extraction
    -------------------------------------------------------
    Parameters:
        dataset_name - HuggingFace dataset identifier (str)
        dataset_config - HuggingFace dataset configuration (str)
        model_name - HuggingFace model identifier (str)
        max_length - Maximum sequence length (int)
        batch_size - Batch size for processing (int)
        device - Device to load model on (str)
        load_in_4bit - Whether to use 4-bit quantization (bool)
        use_flash_attention - Whether to use flash attention (bool)
        split - Dataset split to use (str)
        text_column - Column containing text data (str)
    -------------------------------------------------------
    """
    dataset_name: str
    dataset_config: str
    model_name: str
    max_length: int = 256
    batch_size: int = 32
    device: str = get_device()
    load_in_4bit: bool = True
    use_flash_attention: bool = True
    split: str = "train"
    text_column: str = "text"


class BaseActivationExtractor(ABC, torch.nn.Module):
    """
    -------------------------------------------------------
    Base class for extracting activations from LLMs
    -------------------------------------------------------
    Parameters:
        config - Configuration for model loading and extraction (ModelConfig)
    -------------------------------------------------------
    """

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config

        log.info(f"Using {config.model_name} extractor on {config.device}")
        # Configure quantization
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, device_map=self.config.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self._load_model()
        self.model.eval()

        # Register hook for activation extraction
        self.activations = None
        self._register_hooks()
        self.batch_norm = None

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """
        -------------------------------------------------------
        Load and prepare the specific LLM architecture
        -------------------------------------------------------
        Returns:
            model - Loaded and prepared model (torch.nn.Module)
        -------------------------------------------------------
        """
        pass

    def _activation_hook(self, layer, layer_inputs, layer_outputs):
        """
        -------------------------------------------------------
        Hook to capture activations from the model's final layer
        -------------------------------------------------------
        """
        self.activations = F.relu(layer_outputs)
        # Get rid of normalization, since we we cannot keep track of it outside training
        # if self.batch_norm is None:
        #     self.batch_norm = torch.nn.BatchNorm1d(self.activations.size(1)).to(self.config.device)

        # # Use batch norm instead
        # self.activations = self.batch_norm(self.activations)
        # log.debug(f'Statistics after: {torch.min(self.activations)=}|{torch.max(self.activations)=}|{torch.mean(self.activations)=}')


    def _register_hooks(self):
        """
        -------------------------------------------------------
        Register forward hooks on the final layer
        -------------------------------------------------------
        """
        final_layer = self._get_final_layer()
        final_layer.register_forward_hook(self._activation_hook)

    @abstractmethod
    def _get_final_layer(self) -> torch.nn.Module:
        """
        -------------------------------------------------------
        Get the final layer of the model before the LM head
        -------------------------------------------------------
        Returns:
            layer - Final transformer layer (torch.nn.Module)
        -------------------------------------------------------
        """
        pass

    @abstractmethod
    def get_activation_dim(self) -> int:
        """
        -------------------------------------------------------
        Get the dimension of the final layer of the model before the LM head
        -------------------------------------------------------
        Returns:
            dim - Final transformer layer output dimension(int)
        -------------------------------------------------------
        """
        pass

    def decode_tokens(self, tokens: torch.Tensor) -> List[str]:
        """
        -------------------------------------------------------
        Decode token IDs to strings
        -------------------------------------------------------
        Parameters:
            tokens - Tensor of token IDs (torch.Tensor)
        Returns:
            token_strings - List of decoded tokens (List[str])
        -------------------------------------------------------
        """
        return self.tokenizer.batch_decode(tokens)

    def extract_activations(self, texts: List[str], top_k: int = 10) -> dict[str, Any]:
        """
        -------------------------------------------------------
        Extract activations from a batch of texts
        -------------------------------------------------------
        Parameters:
            texts - batch list of input texts (List[str])
            top_k - Number of top tokens to extract (int)
        Returns:
            results
                activations - Extracted and ReLU'd activations (torch.Tensor)
                top_token_probs - Top k token probabilities (torch.Tensor)
                top_tokens - Top k token indices (torch.Tensor)
                logits - Logits from the model (torch.Tensor)
        -------------------------------------------------------
        """

        inputs = self.tokenizer(
            texts,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            # Get both hidden states (via hook) and logits
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Get top k tokens and their probabilities
            probs = F.softmax(logits, dim=-1)
            top_probs, top_tokens = torch.topk(probs, k=top_k, dim=-1)
        log.debug(f"Extracted activations with {self.activations.shape}")
        return {
            'activations': self.activations,
            'top_token_probs': top_probs,
            'top_tokens': top_tokens,
            'logits': logits
        }


class GPT2ActivationExtractor(BaseActivationExtractor):
    """
    -------------------------------------------------------
    Activation extractor for GPT-2 models
    -------------------------------------------------------
    """

    def _load_model(self) -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=self.quantization_config,
            device_map=self.config.device
        )
        return prepare_model_for_kbit_training(model)

    def _get_final_layer(self) -> torch.nn.Module:
        return self.model.transformer.ln_f

    def get_activation_dim(self) -> int:
        return self._get_final_layer().normalized_shape[0]



class LlamaActivationExtractor(BaseActivationExtractor):
    """
    -------------------------------------------------------
    Activation extractor for LLaMA models
    -------------------------------------------------------
    """

    def _load_model(self) -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=self.quantization_config,
            device_map=self.config.device,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else "sdpa"
        )
        return prepare_model_for_kbit_training(model)

    def _get_final_layer(self) -> torch.nn.Module:
        return self.model.model.norm

    def get_activation_dim(self) -> int:
        return self._get_final_layer().weight.shape[0]


class GemmaActivationExtractor(BaseActivationExtractor):
    """
    -------------------------------------------------------
    Activation extractor for Gemma models
    -------------------------------------------------------
    """

    def _load_model(self) -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=self.quantization_config,
            device_map=self.config.device,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else "sdpa"
        )
        return prepare_model_for_kbit_training(model)

    def _get_final_layer(self) -> torch.nn.Module:
        return self.model.model.norm

    def get_activation_dim(self) -> int:
        return self._get_final_layer().weight.shape[0]


class TextDataset(torch.utils.data.Dataset):
    """
    -------------------------------------------------------
    Dataset class for managing text data
    -------------------------------------------------------
    Parameters:
        tokenizer - Tokenizer for the dataset (PreTrainedTokenizerFast)
        config - Configuration for dataset loading (DataConfig)
    -------------------------------------------------------
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerFast,
            config: DataConfig,
    ):
        # Load dataset (English only and stream for big datasets)
        self.dataset = load_dataset(
            config.dataset_name, name=config.dataset_config, 
            split=config.split, # streaming=True
        ).select_columns([config.text_column])
        self.dataset = self.dataset.shuffle(buffer_size=1000)
        self.config = config
        self.tokenizer = tokenizer

    def get_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        """
        -------------------------------------------------------
        Create DataLoader for extracting activations
        -------------------------------------------------------
        Parameters:
            batch_size - Batch size for processing (int)
            shuffle - whether to shuffle the data (bool)
        Returns:
            dataloader - DataLoader yielding batches of texts (DataLoader)
        -------------------------------------------------------
        """
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
        )
