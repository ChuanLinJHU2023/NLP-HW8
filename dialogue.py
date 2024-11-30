from __future__ import annotations
import itertools
from typing import Dict, Tuple, List

class Dialogue(Tuple[Dict[str,str]]):
    """A dialogue among multiple speakers, represented as an imutable tuple of
    dialogue turns. Each turn is a dict with 'speaker' and 'content' keys. The
    speaker values are just names like "teacher" and "student", or "Alice" and
    "Bob".
    
    See `agents.py` for classes that will extend the Dialogue using an LLM.
    """
    
    def __repr__(self) -> str:
        # Invoked by the repr() function, and also by print() and str() functions since we haven't defined __str__.
        return '\n'.join([f"({turn['speaker']}) {turn['content']}" for turn in self])
    
    def __rich__(self) -> str:
        # Like __str__, but invoked by rich.print().
        return '\n'.join([f"[white on blue]({turn['speaker']})[/white on blue] {turn['content']}" for turn in self])

    def __format__(self, specification: str) -> str:
        # Like __str__, but invoked by f"..." strings and the format() function.
        # We will ignore the specification argument.
        return self.__rich__()
    
    def script(self) -> str:
        """Return a single string that formats this dialogue like a play script,
        suitable for inclusion in an LLM prompt."""
        return '"""\n' + '\n\n'.join([f"{turn['speaker']}: {turn['content']}" for turn in self]) + '\n"""'

    def add(self, speaker: str, content: str) -> Dialogue:
        """Non-destructively append a given new turn to the dialogue."""
        return Dialogue(itertools.chain(self, ({'speaker': speaker, 'content': content},)))
    
    def rename(self, old: str, new: str) -> Dialogue:
        """Non-destructively rename a speaker in a dialogue."""
        d = Dialogue()
        for turn in self:
            d = d.add(new if turn['speaker']==old else turn['speaker'], turn['content'])
        return d
    
    # Support +,  *, and [] operators to concatenate and slice Dialogues.
    # This could be useful when constructing your own argubots.
    
    def __add__(self, other):
        if not isinstance(other, Dialogue):
            raise ValueError(f"Can only concatenate Dialogues with Dialogues, but got {type(other)}")
        return Dialogue(super().__add__(other))
    
    def __mul__(self, other):
        return Dialogue(super().__mul__(other))
    
    def __rmul__(self, other):
        return Dialogue(super().__rmul__(other))
    
    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(index, slice):
            return Dialogue(result)
        else:
            return result
