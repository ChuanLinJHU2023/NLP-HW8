"""This module contains argument bots. 
These agents should be able to handle a wide variety of topics and opponents.
They will be evaluated using methods in `evaluate.py`.
We've included a few to get your started."""

import logging
from rich.logging import RichHandler
from pathlib import Path
import random
import glob
from dialogue import Dialogue
from agents import Agent, ConstantAgent, LLMAgent
from kialo import Kialo

# Use the same logger as agents.py, since argubots are agents;
# we split this file 
# You can change the logging level there.
log = logging.getLogger("agents")    

#############################
## Define some basic argubots
#############################

# Airhead (aka Absentia or Acephalic) always says the same thing.

airhead = ConstantAgent("Airhead", "I know right???")

# Alice is a basic prompted LLM.  You are trying to improve on Alice.
# Don't change the prompt -- instead, make a new argubot with a new prompt.

alice = LLMAgent("Alice",
                 system="You are an intelligent bot who wants to broaden your user's mind. "
                        "Ask a conversation starter question.  Then, WHATEVER "
                        "position the user initially takes, push back on it. "
                        "Try to help the user see the other side of the issue. "
                        "Answer in 1-2 sentences. Be thoughtful and polite.")

############################################################
## Other argubot classes and instances -- add your own here! 
############################################################

class KialoAgent(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""
    
    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo
                
    def response(self, d: Dialogue) -> str:

        if len(d) == 0:   
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            previous_turn = d[-1]['content']  # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")
            
            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])
        
        return claim    
    
# Akiko doesn't use an LLM, but looks up an argument in a database.
  
akiko = KialoAgent("Akiko", Kialo(glob.glob("data/*.txt")))   # get the Kialo database from text files


###########################################
# Define your own additional argubots here!
###########################################


class KialoAgent2(Agent):
    """ KialoAgent subclasses the Agent class. It responds with a relevant claim from
    a Kialo database.  No LLM is used."""

    def __init__(self, name: str, kialo: Kialo):
        self.name = name
        self.kialo = kialo

    def response(self, d: Dialogue) -> str:

        if len(d) == 0:
            # First turn.  Just start with a random claim from the Kialo database.
            claim = self.kialo.random_chain()[0]
        else:
            # previous_turn = d[-1]['content']  # previous turn from user
            previous_turn = ";".join([ d[i]['content']*(i+1) for i in range(len(d)) ] ) # previous turn from user
            # Pick one of the top-3 most similar claims in the Kialo database,
            # restricting to the ones that list "con" arguments (counterarguments).
            neighbors = self.kialo.closest_claims(previous_turn, n=3, kind='has_cons')
            assert neighbors, "No claims to choose from; is Kialo data structure empty?"
            neighbor = random.choice(neighbors)
            log.info(f"[black on bright_green]Chose similar claim from Kialo:\n{neighbor}[/black on bright_green]")

            # Choose one of its "con" arguments as our response.
            claim = random.choice(self.kialo.cons[neighbor])

        return claim

    # Akiko doesn't use an LLM, but looks up an argument in a database.


akiki = KialoAgent2("Akiki", Kialo(glob.glob("data/*.txt")))  # get the Kialo database from text files


###########################################
from agents import *

def kialo_responses(kialo: Kialo, s: str) -> str:
    c = kialo.closest_claims(s, kind='has_cons')[0]
    result = f'One possibly related claim from the Kialo debate website:\n\t"{c}"'
    if kialo.pros[c]:
        result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users in favor of that claim:"] + kialo.pros[c])
    if kialo.cons[c]:
        result += '\n' + '\n\t* '.join(["Some arguments from other Kialo users against that claim:"] + kialo.cons[c])
    return result


class RAG_Agent(LLMAgent):

    def response(self, d: Dialogue, kialo: Kialo, **kwargs) -> str:
        """Ask the LLM how it would continue the dialogue."""
        # kialo = Kialo(glob.glob("data/*"))

        messages = dialogue_to_openai(d, speaker=self.name, **self.kwargs_format)

        pretty_messages = '\n'.join([f"[black on bright_yellow]({m['role']})"
                                     f"[/black on bright_yellow] {m['content']}" for m in messages])
        pretty_kws = " with " + ", ".join(
            f"{key}={val}" for key, val in self.kwargs_llm.items()) if self.kwargs_llm else ""
        log.info(f"Calling LLM {self.model}{pretty_kws}:\n{pretty_messages}")
        ##### NEXT LINE IS WHERE THE MAGIC HAPPENS #####
        response = self.client.chat.completions.create(messages=messages,
                                                       model=self.model, **(self.kwargs_llm | kwargs))
        # kwargs passed to this response() call override those passed to __init__()
        log.debug(f"Response from LLM:\n[black on white]{response}[/black on white]")

        # That's it - now we have our response!  Get the content out of it.

        choice: chat.chat_completion.Choice = response.choices[0]
        content = choice.message.content
        if not isinstance(content, str):
            raise ValueError("No content string returned from {self.kwargs_llm['client']}")

        # Clean up the returned content a little bit.

        if choice.finish_reason == 'length':
            # indicate that response was cut off due to max_tokens
            content += " ..."

        speaker = f"{self.name}: "
        if content.startswith(speaker):
            # Generated response was unfortunately in the form "Alice: I agree with you."
            # Remove the "Alice: " part.
            # (This could happen if the messages you sent to the LLM included speaker names,
            # for example if you called `dialogue_to_openai` with speaker_names=True.)
            content = content[len(speaker):]

            # Log the content part of the LLM's response, but only if
        # we didn't already log the whole thing above.
        if log.getEffectiveLevel() > logging.DEBUG:
            log.info(f"Response from LLM:\n[black on white]{content}[/black on white]")

        return content
