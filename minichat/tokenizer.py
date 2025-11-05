import os
import copy
from functools import lru_cache


SPECIAL_TOKENS = [
    "<|bos|>", # Beginning of Sequence
    "<|user_start|>",
    "<|user_end|>",   # User Input Delimiters
    "<|assistant_start|>",
    "<|assistant_end|>", # Assistant Response Delimiters
    "<|python_start|>",
    "<|python_end|>",   # Python Code Delimiters
    "<|output_start|>",
    "<|output_end|>",   # Output Delimiters
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers import pre_tokenizers, decoders, regex
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(cls, path):
        tokenizer = HFTokenizer.from_pretrained(path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)
    
    @classmethod
    def train_friom_iterator(cls, text_iterator, vocab_size):
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,
            unk_token=None,
            fuse_unk=False
        ))
        tokenizer.normalizer = None
        gpt4_split_regex = regex.Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = None
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = 
    
    
