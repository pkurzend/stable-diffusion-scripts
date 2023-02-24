import inspect
import re
from typing import Callable, List, Optional, Union

import numpy as np


import inspect
from typing import Callable, List, Optional, Union

import torch

import random

from diffusers import StableDiffusionPipeline

# inspiration from here: https://huggingface.co/waifu-research-department/long-prompt-weighting-pipeline



re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res

# not used!
def tokenize_weighted_text(tokenizer, text):
    result = {"input_ids" : [], "attention_mask" : []}
    start_id = tokenizer.convert_tokens_to_ids('<|startoftext|>')
    end_id = tokenizer.convert_tokens_to_ids('<|endoftext|>')

    for chunk, weight in parse_prompt_attention(text):
        tokenized_chunk = tokenizer(chunk)
        
        input_ids = tokenized_chunk["input_ids"][1:-1] # remove end token and start tokens
        attention_mask = tokenized_chunk["attention_mask"][1:-1]

        print(chunk, input_ids, attention_mask, weight)

        result["input_ids"] += input_ids
        result["attention_mask"] += [a*weight for a in attention_mask]

    result["input_ids"] = [start_id] + result["input_ids"] + [end_id]
    result["attention_mask"] = [1] + result["attention_mask"] + [1]

    return result

def get_token_weights(tokenizer, text):
    attention_masks = []
    
    return_list = True
    if isinstance(text, str):
        text = [text]
        return_list = False

    prompts = []

    for promp in text:
        clean_prompt = ""
        prompt_attention_mask = []

        for chunk, weight in parse_prompt_attention(promp):
            clean_prompt += chunk
            tokenized_chunk = tokenizer(chunk)
          
            attention_mask = tokenized_chunk["attention_mask"][1:-1]

            prompt_attention_mask += [a*weight for a in attention_mask]

        prompt_attention_mask = [1] + prompt_attention_mask #+ [1] # adding 1 for start and end token

        attention_masks.append(prompt_attention_mask)
        prompts.append(clean_prompt)

    if return_list:
        prompts = prompts
    else:
        prompts = prompts[0]
    return torch.Tensor(attention_masks), prompts


def _encode_prompt(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=None, negative_prompt_embeds=None):
    r"""
    Encodes the prompt into text encoder hidden states.
    Args:
        prompt (`str` or `list(int)`):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_images_per_prompt (`int`):
            number of images that should be generated per prompt
        do_classifier_free_guidance (`bool`):
            whether to use classifier free guidance or not
        negative_prompt (`str` or `List[str]`):
            The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
            if `guidance_scale` is less than `1`).
    """
    batch_size = len(prompt) if isinstance(prompt, list) else 1

    token_weights, prompt = get_token_weights(self.tokenizer, prompt)
    print(prompt)


    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # print(text_inputs)

    text_input_ids = text_inputs.input_ids


    # text_embedding = self.text_encoder(text_input_ids)[0]
    # print(text_embedding)
    

    untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
        print(
            "The following part of your input was truncated because CLIP can only handle sequences up to"
            f" {self.tokenizer.model_max_length} tokens: {removed_text}"
        )

    if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None

    text_embeddings = self.text_encoder(
        text_input_ids.to(device),
        attention_mask=attention_mask,
    )
    text_embeddings = text_embeddings[0]


    
    print(token_weights)
    for i in range(min(token_weights.shape[1], 77)):
        text_embeddings[0][i] = text_embeddings[0][i] * token_weights[0][i]
        # maybe scale the embedding (divide by mean)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
    text_embeddings = text_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        token_weights, uncond_tokens = get_token_weights(self.tokenizer, uncond_tokens)

        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = uncond_input.attention_mask.to(device)
        else:
            attention_mask = None

        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(device),
            attention_mask=attention_mask,
        )
        uncond_embeddings = uncond_embeddings[0]

        
        print(token_weights)
        for i in range(min(token_weights.shape[1], 77)):
            uncond_embeddings[0][i] = uncond_embeddings[0][i] * token_weights[0][i]
            # maybe scale the embedding (divide by mean)

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = uncond_embeddings.shape[1]
        uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
        uncond_embeddings = uncond_embeddings.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    return text_embeddings



def add_prompt_weighting(pipe):
    pipe._encode_prompt = _encode_prompt.__get__(pipe, StableDiffusionPipeline)
    return pipe