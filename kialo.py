# CS465 at Johns Hopkins University.
# Author: Jason Eisner <jason@cs.jhu.edu>, November 2023

"""
A simple module for navigating a collection of discussion trees exported from kialo.com.
"""

from collections import defaultdict
from functools import cache
import random
import re
from rank_bm25 import BM25Okapi as BM25_Index 
from typing import Dict, Tuple, List, Iterable, Callable, Optional

        
def tokenize_simple(s: str) -> List[str]:
    """Simple tokenizer that can be used with BM25."""
    s = s.lower()                 # lowercase
    s = re.sub(r'[^\w\s]', '', s)  # remove punctuation
    return s.split(" ")  


class Kialo:
    
    """A collection of natural-language claims, arranged into trees."""
        
    class Claim(str):
        """A claim that appears in Kialo discussions."""  
        # In the original file format, each claim can be uniquely identified by
        # a discussion number and a claim id within that discussion.  However,
        # to simplify the code and the interface, we'll instead identify the
        # claim with its actual text.  Thus, Claim is just a subtype of str.
        # Claims with identical text cannot be distinguished."""
        pass
    
    def __init__(self, filenames: List[str], tokenizer: Optional[Callable[[str], List]] = tokenize_simple):
        """The files should have been exported from kialo.com.
        The tokenizer is allowed to do splitting at whitespace, stemming, BPE, integerization, etc."""
        
        # Each node is a Claim that stores one parent (or None) and lists of pro
        # and con children.  
        #
        # Note: The same Claim could appear multiple times because of
        # cross-referencing within a file, or even potentially because of
        # textual duplication within or across files.  In these (rare) cases,
        # the Claim will store only the *first* of the parents we found for it,
        # but will store all of the children from all occurrences of the Claim.
        # There can be descendant cycles but not ancestor cycles.

        self.roots: List[Kialo.Claim] = []                                   # root claims of the discussion trees
        self.parents: Dict[Kialo.Claim, Optional[Kialo.Claim]] = {}          # maps each claim to its parent, or None in the case of a root
        self.pros: Dict[Kialo.Claim, List[Kialo.Claim]] = defaultdict(list)  # maps each claim to its "pro" children (initially [])
        self.cons: Dict[Kialo.Claim, List[Kialo.Claim]] = defaultdict(list)  # maps each claim to its "con" children (initially [])
        self.claims: Dict[str, List[str]] = {}                               # maps a kind of claim to the list of all claims of that kind
        self.bm25: Dict[str, BM25_Index] = {}                                # maps a kind of claim to a BM25 index on claims of that kind
        self.tokenizer = tokenizer                                           # tokenizer for use with BM25 index
        
        assert isinstance(filenames, List), "Argument should be a list of filenames"
        for filename in filenames:
            self.add_discussion(filename)
            
    def __len__(self) -> int:
        """Number of claims in the data structure."""
        return len(self.parents)
    
    def add_discussion(self, filename: str):
        """Add a tree of claims from an exported Kialo discussion.
        This method has to parse the file format."""
        
        # Some variables are invalidated when we add new claims.
        # However, depthmemo is not, because the depths of existing claims won't change.
        self.bm25 = {}   # flush current index of claims, since we're going to add new claims
        self.claims = {}

        # Each claim in the file appears on a pair of lines, where the first
        # gives metadata and the second gives the claim itself.  From the
        # claim's id, we can compute its parent's id.  (We use only the ids to
        # get the structure of the tree, ignoring the fact that a claim's
        # subtrees are indented immediately below the claim.)

        index: Dict[str, Kialo.Claim] = {}   # map claim id to actual claim; ids will be discarded after reading file
        metadata_re = re.compile('(((\\d+\\.)*)\\d+.)\\s+(Thesis|Pro|Con):')  # matches metadata format
        xref_re = re.compile('-> See ((\\d+\\.)+)')                           # matches cross-reference format
        with open(filename, encoding='utf-8') as f:
            id, id_parent, polarity = None, None, None   # vars maintained throughout loop below
            for line in f:
                line = line.strip()                      # remove leading and trailing whitespace
                match = metadata_re.match(line)
                if match:     # this line gives metadata that we'll remember for a claim on the next line
                    id, id_parent, polarity = match.group(1), match.group(2), match.group(4)
                    if not (id_parent=='') == (polarity=='Thesis'):
                        raise ValueError(f"Thesis should appear at root of tree and nowhere else: {line}")
                else:
                    if id is not None:   # previous line gave us metadata about a claim
                        # this line is the actual claim
                
                        # if claim is a cross-reference, substitute the text of the referenced claim
                        match = xref_re.match(line)
                        if match:
                            line = index[match.group(1)]  # will raise KeyError if referenced claim didn't appear earlier in file
                        else:
                            # remove any bracketed footnote numbers or footnoted page refs
                            line = re.sub(' \\(p[^)]* \\[\\d+\\]\\)', '', line)
                            line = re.sub(' \\[\\d+\\]', '', line)
                        
                        index[id] = claim = Kialo.Claim(line)    # remember it by numeric id for later lookup while processing this file
                        
                        # link the claim into the graph.  In the rare case where we found
                        # another copy of this claim before, we will take care not to replace
                        # the stored parent, but we will still add this claim as a child of
                        # its newly found parent.
                        if polarity == 'Thesis':
                            if claim not in self.parents:
                                self.parents[claim] = None
                            self.roots.append(claim)
                        else:
                            # assertions below hold because previous line had metadata
                            assert id_parent is not None
                            parent = index[id_parent]
                            if claim not in self.parents:
                                self.parents[claim] = parent
                            if polarity == 'Pro':
                                self.pros[parent].append(claim)
                            else:
                                assert polarity == 'Con'
                                self.cons[parent].append(claim)
                    id, id_parent, polarity = None, None, None   # this line did not give metadata
                    
    
    @cache
    def depth(self, claim: 'Kialo.Claim'):
        """Depth of claim in the tree, nonstandardly counting the root as depth 1."""

        # The @cache decorator memoizes the function.  Thus, calling depth() on
        # *all* claims will take only linear total time.
        # 
        # Note that we don't have to flush the memo table when calling
        # `add_discussion()`, because that method is careful not to change the
        # depth of existing claims.
        
        parent = self.parents[claim]
        if parent is None:
            return 1
        else:
            return 1 + self.depth(parent)
    
    
    def random_chain(self, n: int = 1):
        """Return a random chain of arguments of length n (or shorter if necessary)."""
        # Returns in linear time.    

        if len(self)==0:
            return []

        maxdepth = max(self.depth(claim) for claim in self.parents)
        if n > maxdepth:  
            n = maxdepth  # reduce ambition
            
        # Choose a sufficiently deep claim to be the final claim
        claim = random.choice([claim for claim in self.parents if self.depth(claim) >= n])

        # Ascend to parents to build up the full chain in reverse.
        chain = []
        while n > 0:   # number of claims left to add
            assert claim is not None
            chain.append(claim)
            n -= 1
            claim = self.parents[claim]

        chain.reverse()
        return chain
 
    
    def closest_claims(self, s: str, n: int = 1, kind: str = 'all') -> List['Kialo.Claim']:
        """Return the n claims of a given `kind` that are most similar to s,
        ranked from most to least similar."""
        
        if self.tokenizer is None:
            raise ValueError("Can't call this method on a Kialo object that doesn't have a tokenizer")
              
        # Build index of claims of the given `kind`, if we don't currently have such an index.
        if kind not in self.claims:
            # first find all claims of this kind
            if kind == 'all':          # all claims
                self.claims[kind] = [claim for claim in self.parents]
            elif kind == 'has_cons':   # all claims that have at least one "con" response
                self.claims[kind] = [claim for claim in self.parents if self.cons[claim]]
            elif kind == 'has_pros':   # all claims that have at least one "pro" response
                self.claims[kind] = [claim for claim in self.parents if self.pros[claim]]
            else:
                raise ValueError(f"Don't know about claims of kind {kind}")
            # now build the index 
            if self.claims[kind]:   # only build if there are some claims, to avoid div by 0
                self.bm25[kind] = BM25_Index(self.claims[kind], tokenizer = self.tokenizer)

        # Do the retrieval.
        if self.claims[kind]:
            return self.bm25[kind].get_top_n(self.tokenizer(s), self.claims[kind], n=n)    
        else:
            return []   # no claims of this kind, and no index



