import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import Regex, pre_tokenizers, decoders
from tokenizers.trainers import BpeTrainer

from minichat.common import get_base_dir

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

gpt4_split_regex = Regex(SPLIT_PATTERN)


class HFTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    @classmethod
    def from_pretrained(cls, path):
        tokenizer = Tokenizer.from_pretrained(path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = Tokenizer(BPE(byte_fallback=True,
                                  unk_token=None,
                                  fuse_unk=False))
        tokenizer.normalizer = None
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior='isolated', invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor=None
        trainer = BpeTrainer(
            vocab_size= vocab_size,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS
        )
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_bos_token_id(self):
        return self.tokenizer.encode_special("<|bos|>")

    def _encode_one(self, text, prepand=None, append=None):
        assert isinstance(text, str)
        ids = []
        if prepand is not None:
            prepand_id = prepand if isinstance(prepand, int) else self.encode_special(prepand)
            ids.append(prepand_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError("Input should be a string or a list of strings.")
    
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, 'tokenizer.json')
        self.tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved to {tokenizer_path}")
    



def get_tokenizer():
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    tokenizer = HFTokenizer.from_pretrained(tokenizer_dir)
    return tokenizer




