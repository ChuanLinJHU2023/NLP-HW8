from __future__ import annotations
from math import nan
import itertools
import os
import dotenv
import json
import pathlib
import openai
from typing import Union, Dict
from contextlib import contextmanager

Usage = Dict[str, Union[int, float]]   # TODO: could have used Counter class
default_usage_file = pathlib.Path("usage_openai.json")

# prices match https://openai.com/api/pricing/ as of November 2024,
# but they don't consider discounts for caching or batching.
# Not all models are listed in this file, nor is the fine-tuning API.
pricing = {   # per 1M tokens
        'gpt-4o-mini':               { 'input':  0.15, 'output':  0.60, }, 
        'gpt-4o':                    { 'input':  2.50, 'output': 10.00, },
        'gpt-3.5-turbo-0125':        { 'input':  0.50, 'output':  1.50, },  # supports the legacy completions API, not chat completions
        'gpt-3.5-turbo-instruct':    { 'input':  1.50, 'output':  2.00, },  # supports the legacy completions API, not chat completions
        'o1-mini':                   { 'input':  3.00, 'output': 12.00, },  # reasoning model for STEM
        'o1-preview':                { 'input': 15.00, 'output': 60.00, },  # reasoning model for STEM
        'text-embedding-3-small':    { 'input':  0.02, 'output':  0.02, },  # supports only the embeddings API, so output tokens aren't used
        'text-embedding-3-large':    { 'input':  0.13, 'output':  0.13, },  # supports only the embeddings API, so output tokens aren't used
    }

# Default models.  These variables can be imported from this module.  
# Even if the system that is being evaluated uses a cheap default_model.
# one might want to evaluate it carefully using a more expensive default_eval_model.
default_model = "gpt-4o-mini"
default_eval_model = "gpt-4o-mini"

# A context manager that lets you temporarily change the default models
# during a block of code.  You can write things like
#     with use_model('gpt-4o'):
#        ...
# 
#     with use_model(eval_model='gpt-4o'):
#        ...
@contextmanager
def use_model(model: str = default_model, eval_model: str = default_eval_model):
    global default_model, default_eval_model
    save_model, save_eval_model = default_model, default_eval_model
    default_model, default_eval_model = model, eval_model
    try:
        yield
    finally:
        default_model, default_eval_model = save_model, save_eval_model        


def track_usage(client: openai.OpenAI, path: pathlib.Path = default_usage_file) -> openai.OpenAI:
    """
    This method modifies (and returns) `client` so that its API calls
    will log token counts to `path`. If the file does not exist it
    will be created after the first API call. If the file exists the new 
    counts will be added to it.  
    
    The `read_usage()` function gets a Usage object from the file, e.g.:
    {
        "completion_tokens": 20,
        "prompt_tokens": 30,
        "total_tokens": 50,
        "cost": 0.00002
    }
    
    >>> client = openai.OpenAI()
    >>> track_usage(client, "example_usage_file.json")
    >>> type(client)
    <class 'openai.OpenAI'>
    
    """
    old_completion = client.chat.completions.create
    def tracked_completion(*args, **kwargs):
        response = old_completion(*args, **kwargs)
        old: Usage = read_usage(path)
        new: Usage = get_usage(response, model=kwargs.get('model',None))
        _write_usage(_merge_usage(old, new), path)
        return response

    old_embed = client.embeddings.create
    def tracked_embed(*args, **kwargs):
        response = old_embed(*args, **kwargs)
        old: Usage = read_usage(path)
        new: Usage = get_usage(response, model=kwargs.get('model',None))
        _write_usage(_merge_usage(old, new), path)
        return response

    client.chat.completions.create = tracked_completion    # type:ignore
    client.embeddings.create = tracked_embed   # type:ignore
    return client

def get_usage(response, model=None) -> Usage:
    """Extract usage info from an OpenAI response."""
    usage: Usage = vars(response.usage).copy()
    try:
        costs = pricing[response.model]  # model name returned in response (may be alias)
    except KeyError:
        try: 
            costs = pricing[model]       # model name passed in request (may be alias)
        except KeyError:
            raise ValueError(f"Don't know prices for model {model} or {response.model}")

    cost = (  usage.get('prompt_tokens',0)     * costs['input']
            + usage.get('completion_tokens',0) * costs['output']) / 1_000_000
    usage['cost'] = cost

    return usage    

def read_usage(path: pathlib.Path = default_usage_file) -> Usage:
    """Retrieve total usage logged in a file."""
    if os.path.exists(path):
        with open(path, "rt") as f:
            return json.load(f)
    else:
        return {}

def _write_usage(u: Usage, path: pathlib.Path):
    with open(path, "wt") as f:
        json.dump(u, f, indent=4)

def _merge_usage(u1: Usage, u2: Usage) -> Usage:
    return {k: u1.get(k, 0) + u2.get(k, 0) for k in itertools.chain(u1,u2)}
     
def new_default_client() -> openai.OpenAI:
    """Set the `default_client` to a new tracked client, based on the current
    `OPENAI_API_KEY` and its current org. Thereafter, everyone can use it as a
    convenience by importing `default_client` from this module. If you change
    your API key or its default org, you should call this method again."""
    global default_client
    dotenv.load_dotenv(override=True)              # set environment variable OPENAI_API_KEY from .env
    default_client = track_usage(openai.OpenAI())  # create a client with default args, and modify it 
                                                   # so that it will store its usage in a local file 
    return default_client

new_default_client()       # set `default_client` right away when importing this module
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
