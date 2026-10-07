"""Pinned model and data adapter used by WAGLE scoring."""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dataset.fair_wmdp import FairWMDPUnlearnDataset
from dataset.Base import unlearncollector


class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tokenizer_name = kwargs.get("tokenizer_name", model_name)
        self.if_llama = "llama" in model_name
        self.fair_pair_stream = kwargs["fair_pair_stream"]
        self.expected_pair_stream_sha256 = kwargs["expected_pair_stream_sha256"]

    def init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id
        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except (AttributeError, KeyError):
            self.device = torch.device("cuda:0")

    def init_dataset(self):
        self.unlearn_dataset = FairWMDPUnlearnDataset(
            self.fair_pair_stream, self.expected_pair_stream_sha256
        )
        self.unlearn_collator = unlearncollector
        self.test_dataset = self.test_collator = None
