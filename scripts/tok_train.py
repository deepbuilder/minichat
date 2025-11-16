import time
import os
import torch
from minichat.tokenizer import HFTokenizer
from minichat.dataloader import data_loader
from minichat.dataset import iter_batched
from minichat.common import get_base_dir

max_doc_len = 10_000
max_chars = 10_000_000_000
vocab_size = 65_536


def text_iterator():
    nchars = 0
    for batch in iter_batched('train'):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > max_doc_len:
                doc_text = doc[:max_doc_len]
            nchars += len(doc_text)
            yield doc_text
            if nchars > max_chars:
                return 

text_iter = text_iterator()

t0 = time.time()
tokenizer = HFTokenizer.train_from_iterator(text_iter, vocab_size)
t1 = time.time()

train_time = t1 - t0


base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, 'tokenizer')
tokenizer.save(tokenizer_dir)

# Sanity check

test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# Cache the token ids

vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_ids = [ix for ix in range(vocab_size)]

sizes = []
for token in token_ids:
    token_string = tokenizer.decode([token])
    if token_string in special_set:
        sizes.append(0)
    else:
        sizes.append(len(token_string.encode("utf-8")))

token_bytes = torch.tensor(sizes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, 'w') as file:
    torch.save(token_bytes, token_bytes_path)
print(f"Saved token bytes to {token_bytes_path}")
