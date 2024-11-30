from __future__ import annotations
from pathlib import Path
import logging
from rich.logging import RichHandler
from typing import Counter, DefaultDict, List, Tuple
from collections import Counter, defaultdict
import itertools
from math import sqrt, nan
from tqdm import tqdm   # type: ignore

from agents import Agent, CharacterAgent, EvaluationAgent, conjunction
from characters import Character, devset as dev_chars
from dialogue import Dialogue
from simulate import simulated_dialogue
from tracking import read_usage

log = logging.getLogger(Path(__file__).stem)    
if not log.hasHandlers():   # avoid adding again when reimporting
    log.addHandler(RichHandler(level="NOTSET", markup=True,   # allows rich text log messages, with [color]s
                               show_time=False, show_level=False))
log.setLevel(logging.WARNING)   # usually WARNING, but you may want to change to INFO or DEBUG to get more output


# A couple of useful constants used later in the file.

research_team = "NLP class students"  # name of the speaker who is ASKING the evaluation questions (may show up in log messages)
default_judge = Character("Judge Wise", [],   # an external observer who is ANSWERING some of those questions
                          "a social scientist who studies and assesses political conversations")   

# We're going to make a lot of evaluations, so we'd better have a convenient way
# to represent and aggregate them.

class Eval():
    """Aggregated results from one or more dialogue evaluations.

    We track the mean and variance of the score on each numerical question
    (ignoring any missing values), and the list of long-form comments for each
    free-form question.
    
    This class is boring from an NLP point of view -- just a utility class. 
    But it is reasonably general; it could handle other questions.
    """
    n: int                        # number of dialogues whose evaluations are recorded here; used only for reporting
    scores: Counter[str]          # total score on each criterion
    counts: Counter[str]          # number of scores contributing to each total   
    squared_scores: Counter[str]  # total squared score on each criterion
    comments: DefaultDict[str, List[Tuple[str,str]]]  # list of (evaluator,comment) pairs for each long-form question

    def __init__(self,
                 comments: dict[str,List[Tuple[str,str]]] = {},    
                 scores: dict[str,int] = {},
                 ### Remaining arguments are for internal calls only
                 n: int = 1,
                 counts: dict[str,int] | None = None,
                 squared_scores: dict[str,int] | None = None,
                ) -> None: 
        self.comments = defaultdict(list)
        for key, val in comments.items():
            self.comments[key] = val

        self.scores = Counter(scores)
        self.n = n

        if counts is None:
            counts = {key: n for key in self.scores}
        self.counts = Counter(counts)
        
        if squared_scores is None:
            assert n <= 1    # must square the individual scores, not sums of scores
            squared_scores = {k: v**2 for k,v in scores.items()}
        self.squared_scores = Counter(squared_scores)

        assert set(scores.keys()) == set(counts.keys())
        assert set(scores.keys()) == set(squared_scores.keys())

            
    def mean(self) -> dict[str,float]:
        """Returns a dictionary with the mean score on each crtierion."""
        return {k: self.scores[k]/self.counts[k] for k in self.scores}
    
    def sd(self) -> dict[str,float]:
        """Returns a dictionary with the sample standard deviation on each crtierion."""
        return {k: (nan if self.counts[k] <= 1
                    else sqrt((self.squared_scores[k] - self.scores[k]**2 / self.counts[k]) 
                              / (self.counts[k]-1)))
                for k in self.scores}
                
    def __len__(self) -> int:   # the len() function
        return self.n
        
    def __repr__(self) -> str:        
        allcomments = [f"Comments from {question} question:\n"
                        + '\n'.join(f"({c[0]}) {c[1]}" for c in commentlist)
                        for question, commentlist in self.comments.items()]
        if allcomments:
            commentstring = '\n\n' + '\n\n'.join(allcomments)
        else:
            commentstring = ''
      
        if len(self) == 1:
            return (f"<Eval of 1 dialogue: {repr(self.mean())}>{commentstring}")
        else:        
            # TODO: print fewer decimal digits
            return (f"<Eval of {len(self)} dialogues: {repr(self.mean())}>"
                    f"\nStandard deviations: {repr(self.sd())}{commentstring}")
        
    def __iadd__(self, other: Eval) -> Eval:   # the += operator
        if not isinstance(other, Eval):
            raise ValueError(f"Can only add Evals to Evals, but got {type(other)}")
        self.scores += other.scores     # sum Counter dictionaries
        self.n += other.n
        self.counts += other.counts
        self.squared_scores += other.squared_scores 
        for key, val in other.comments.items():
            self.comments[key] += val   # destructively append lists
        return self

    def __add__(self, other: Eval) -> Eval:   # the + operator
        if not isinstance(other, Eval):
            raise ValueError(f"Can only add Evals to Evals, but got {type(other)}")
        comments = defaultdict(list)  # collect all comments here
        for key, val in itertools.chain(self.comments.items(), other.comments.items()):
            comments[key] += val   # append lists
        return Eval(comments = comments,
                    scores = self.scores + other.scores,
                    n = self.n + other.n,
                    counts = self.counts + other.counts,
                    squared_scores = self.squared_scores + other.squared_scores)      


# The prompt text is hardcoded into the two top-level functions below and the
# EvaluationAgent class.
#
# That's easiest to understand, and it's okay for this homework, because the
# evaluation metric is fixed and you don't need any flexibilty to change it.
# 
# But if you were trying to engineer the evaluation scheme to agree with real
# human evaluations, you would want to create many different evaluation objects
# and loop over them.

# Separate issue with the evaluation design:
# 
# Unfortunately, in the functions below, the instructions and the dialogue
# reveal the evaluee's real name.  What the evaluator gives Airhead a lower
# rating just because it's named Airhead?  Or gives Aragorn a higher rating just
# because the evaluator is an LOTR fan?
# 
# We want the evaluator to focus on the _content_ of the dialogue and not be
# unfairly biased by the evaluee's name.  So we should really replace the latter
# by a standard pseudonym throughout the prompt.  But we skipped this step for
# the homework because it would make the log messages harder for you to read.

def eval_by_participant(participant: Character,
                        evaluee: str, dialogue: Dialogue) -> Eval:
    """Ask a `participant` from this `dialogue` what they now feel about 
    the `evaluee` participant (who is usually an argubot).  Inside this method,
    we will instruct `participant` by turning them into an `EvaluationAgent`."""
    name = participant.name
    speakers = {turn['speaker'] for turn in dialogue}
    if not {name, evaluee} <= {turn['speaker'] for turn in dialogue}:
        raise ValueError(f"{name} and {evaluee} did not both participate in dialogue")

    # We're going to start a new dialogue, `d`, with `agent`, to discuss
    # the existing `dialogue`.
    d = Dialogue()
    agent = EvaluationAgent(participant)
  
    # Let's start out with an open-ended warmup question, which serves as a kind
    # of "chain of thought" prompting by raising relevant issues to help with
    # later questions.
    
    warmup = (f"Hello {name}!  Here is a conversation that you had "
              f"with {conjunction(speakers - {name}, zeroval='yourself')}."
              f"\n\n{dialogue.script()}"
              f"\n\nWhat did {evaluee} disagree with you about? How did the conversation go, in your opinion? "
              f"Where could {evaluee} have done better?")
    d = agent.ask(d, research_team, warmup)
    comments = {'overview': [(participant.name, d[-1]['content'])]}
    
    # Now let's get some ratings.
    # Each of these involves temporarily extending that dialogue by another question.
    # The agent does not see the answers to the previous ratings questions, only to the warmup question.
    # (That's easier to arrange with simulated humans than with real ones!)
    
    scores = {}
    question = f"Did {evaluee} listen carefully to you and engage with what you were saying?"
    try: scores['engaged'] = agent.rating(d, research_team, question, 1, 5)  
    except ValueError: pass   # will happen if LLM didn't return an integer in the range 1-5

    for quality in ['informed', 'intelligent', 'moral']:
        question = f"Do you think that people who think like {evaluee} about that topic are {quality}?"
        try: scores[quality] = agent.rating(d, research_team, question, 1, 5)  
        except ValueError: pass
        
    return Eval(comments, scores)


def eval_by_observer(observer: Character, evaluee: str, dialogue: Dialogue) -> Eval:
    """Ask an external observer what they thought about the participant `evaluee` 
    in the given `dialogue` (who is usually an argubot).  Inside this method,
    we will instruct `observer` by turning them into an `EvaluationAgent`."""
    
    # The structure of this function is similar to `eval_by_participant`.
    # We'll use a single score of 1-10 for the main question, rather than
    # breaking it down into subscores.
    
    speakers = {turn['speaker'] for turn in dialogue}
    d = Dialogue()
    agent = EvaluationAgent(observer)
    warmup = (f"Here is a conversation that you observed among {conjunction(speakers)}."
               f"\n\n{dialogue.script()}"
               f"\n\n{evaluee} was trying to make this a constructive converation, "
               f"and to help {conjunction(speakers - {evaluee}, zeroval='themself')} appreciate other points of view. " 
               f"What new perspectives did {evaluee} offer to them?  Was this successful?") 
    d = agent.ask(d, research_team, warmup)
    comments = {'mindopening': [(observer.name, d[-1]['content'])]}
    
    scores = {}
    question = f"How skilled is {evaluee} at helping others become more open-minded?"
    try: scores['skilled'] = agent.rating(d, research_team, question, 1, 10)
    except ValueError: pass
    
    return Eval(comments, scores)

def eval_dialogue(participant: Character,
                  evaluee: str, judge: Character, 
                  dialogue: Dialogue) -> Eval:
    """Combines `eval_by_particpant` and `eval_by_observer` into 
    a single Eval that also includes a summary score."""
    e = (eval_by_participant(participant, evaluee, dialogue)  # creates EvaluationAgent(participant)
         + eval_by_observer(judge, evaluee, dialogue))        # creates EvaluationAgent(judge)

    # Simply add all the scores to get the summary score.
    # In real life, one might use a _weighted_ sum.
    total = sum(e.scores.values())   
    e += Eval(scores={'TOTAL': total})

    assert e.n == 3   # we summed 3 evals so e thinks there were 3 dialogues (maybe we should have specified n=1/3 for each!)
    e.n = 1           # but actually they all came from the same dialogue 
    return e
    
        
        
# We'll store the expensively generated raw data in dictionaries, rather than
# throwing it away once we have a final score.  So you are free to examine or
# replace parts of it in the notebook.

saved_dialogues = {}   # maps argubot name to a list of (dialogue, eval) pairs
saved_evalsum   = {}   # maps argubot name to the sum of all those evals
        
def eval_on_characters(argubot: Agent, 
                       chars: List[Character] = dev_chars, 
                       judge: Character = default_judge,
                       turns: int = 8,
                       reps: int = 2) -> Eval:
    """Evaluate a given argubot against a whole set of Characters.
    
    Return the aggregate evaluation.  Also, store the list of the (dialogue,
    evaluation) pairs in `saved_dialogues[argubot.name]`, where
    `saved_dialogues` is a top-level attribute of the evaluate module. 

    To simplify things for the caller, this method does not currently accept
    arbitrary CharacterAgents and EvaluationAgents.  Rather, for each
    character in Characters, it creates one of each using standard parameters.
    It also creates an EvaluationAgent from the `judge` Character.
    """
    
    # Prepare to keep track of the raw data.
    if argubot.name in saved_dialogues: del saved_dialogues[argubot.name]
    if argubot.name in saved_evalsum:   del saved_evalsum[argubot.name]
    de_list = []
    e_sum   = Eval(n=0)   # empty set
    starting_cost = read_usage()['cost']

    # Do the eval.
    # itertools.product gives us a single iterator for the tqdm progress bar,
    # which is equivalent to the nested loops "for char in chars: for _ in range(reps):"
    for char, _ in tqdm(itertools.product(chars, range(reps)), total=len(chars)*reps): 
        # have the argubot behave
        d = simulated_dialogue(argubot, CharacterAgent(char), turns)
        log.info(d)   # show the final dialogue
        
        # evaluate its behavior
        e = eval_dialogue(char, argubot.name, judge, d)
        log.info(e)   # show the result of evaluation
        
        # add to the growing local record
        de_list.append((d,e))
        e_sum += e

    # We computed all the raw data without any interupts or exceptions.
    # So we can safely save it.
    saved_dialogues[argubot.name] = de_list
    saved_evalsum[argubot.name] = e_sum
    ending_cost = read_usage()['cost']
    log.warning(f"You just spent ${(ending_cost - starting_cost):.2f} of NLP money to evaluate {argubot}")
        
    saved_evalsum[argubot.name] = e_sum
    return e_sum
