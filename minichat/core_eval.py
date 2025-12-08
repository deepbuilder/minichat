import random
import torch
from jinja2 import Template
import torch.distributed as dist


def render_prompt_mc(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{example.query}}{{ continuation_delimiter }}{{example.choices[example.gold]}}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{choice}}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples if fewshot_examples is not None else []
    contexts = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item,
    }
    prompts = [template.render(choice=choice, **contexts) for choice in item['choices']]
    return prompts

def render_prompt_schema(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{context}}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    params = {
        'item': item,
        'continuation_delimiter': continuation_delimiter,
        'fewshot_examples': fewshot_examples if fewshot_examples is not None else [],
    }
    prompts = [template.render(context=context, **params) for context in item['context_options']]
    return prompts

def render_prompt_lm(item, continuation_delimiter, fewshot_examples=None):
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    params = {
        'item': item,
        'continuation_delimiter': continuation_delimiter,
        'fewshot_examples': fewshot_examples if fewshot_examples is not None else [],
    }
    prompt_without = template.render(include_continuation=False, **params)
    prompt_with = template.render(include_continuation=True, **params)
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]

def find_common_length(sequences, direction='prefix'):
    if direction == 'prefix':
        min_len = min(len(seq) for seq in sequences)
        common_len = 0
        for i in range(min_len):
            token = sequences[0][i]
            if all(seq[i] == token for seq in sequences):
                common_len += 1
            else:
                break
        return common_len
    elif direction == 'suffix':
        min_len = min(len(seq) for seq in sequences)
        common_len = 0
        for i in range(1, min_len + 1):
            token = sequences[0][-i]
            if all(seq[-i] == token for seq in sequences):
                common_len += 1
            else:
                break
        return common_len
    else:
        raise ValueError("direction must be 'prefix' or 'suffix'")

def batch_sequences_mc(prompts, tokenizer):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='prefix')
    start_ids = [answer_start_idx] * len(prompts)
    end_ids = [len(t) for t in tokens]
    return tokens, start_ids, end_ids

def batch_sequences_schema(prompts, tokenizer):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    answer_start_idx = find_common_length(tokens, direction='suffix')
    start_ids = [len(t) - answer_start_idx for t in tokens]
    end_ids = [len(t) for t in tokens]
    return tokens, start_ids, end_ids

def batch_sequences_lm(prompts, tokenizer):
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    return [tokens_with], [start_idx], [end_idx]

def stack_sequences(token_lists, pad_token_id):
    max_len = max(len(t) for t in token_lists)
    batch_size = len(token_lists)
    batch_tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    for i, t in enumerate(token_lists):
        batch_tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return batch_tokens

@torch.no_grad()
def forward_model(model, batch_tokens):
    batch_size, seq_len = batch_tokens.shape
    logits = model(batch_tokens)
    target_ids = torch.roll(batch_tokens, -1, dims=1)
    losses = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), target_ids.view(-1), reduction='none'
    ).view(batch_size, seq_len)
    losses[:, -1] = float('nan')
    predictions = torch.argmax(logits, dim=-1)
    return losses, predictions

@torch.no_grad()
def evaluate_example(idx, data, task_meta, model, tokenizer, device):
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(42 + idx)  # Seed with a combination of a constant and the example index
        available_indices = [i for i in range(len(data)) if i != idx]
        # Ensure we don't try to sample more examples than available
        num_fewshot = min(num_fewshot, len(available_indices))
        if num_fewshot > 0:
            fewshot_indices = rng.sample(available_indices, num_fewshot)
            fewshot_examples = [data[i] for i in fewshot_indices]
    
    if task_type == 'multiple_choice':
        prompts = render_prompt_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(prompts, tokenizer)
    elif task_type == 'schema':
        prompts = render_prompt_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(prompts, tokenizer)
    elif task_type  == 'language_modeling':
        prompts = render_prompt_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(prompts, tokenizer)

    if hasattr(model, 'config') and hasattr(model.config, 'sequence_len'):
        max_tokens = model.config.sequence_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(max(0, s - num_to_crop))
                new_end_idxs.append(max(0, e - num_to_crop))
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs
    
    pad_token_id = tokenizer.get_bos_token_id()
    batch_tokens = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, batch_tokens)

    if task_type == 'language_modeling':
        start_idx = start_idxs[0]
        end_idx = end_idxs[0]
        target_ids = batch_tokens[0, start_idx:end_idx]
        pred_ids = predictions[0, start_idx-1:end_idx-1]
        is_correct = torch.all(pred_ids == target_ids).item()
    return is_correct


def evaluate_task(data, task_meta, model, tokenizer, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    size = dist.get_world_size() if dist.is_initialized() else 1

    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for i in range(rank, len(data), size):
        is_correct = evaluate_example(i, data, task_meta, model, tokenizer, device)
        correct[i] = float(is_correct)
    if size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)      
    mean_correct = correct.mean().item()
    return mean_correct