"""
Classes to represent problems as a tree. Very useful for SPoC

Author: Colin Rioux
"""
import typing

class PSPair():
    """
    Pseudo-code, Source code Pair
    
    @property owner the annotator
    @property pid the problem id
    @property the indent within cid
    @property ps a string of pseudocode
    @property sc a string of source code
    """
    def __init__(self, owner: int, pid: str, indent: int, ps: str, sc: str):
        self.owner = owner
        self.pid = pid
        self.indent = indent,
        self.ps = ps
        self.sc = sc

class Node():
    """
    A node within the problem tree
    
    @property data the PSPair for a certain node
    @property parent the parent of the node
    @property children a list of children for the node
    """
    def __init__(self, pspair: PSPair, parent: Node, children: List[Node]) -> Node:
        self.data = pspair
        self.parent = parent
        self.children = children

class Problem():
    """
    An AST-like structure to represent a piece of code
    Each level of the tree corresponds to a particular indent
    If a node has 1<= children, traverse down as we have a block, which is read sequentially by the compiler
    
    @property id the problem id
    @property root the root node of the tree (first line of code)
    """
    def __init__(self, pid: str, root: Node):
        self.id = pid
        self.root = root
