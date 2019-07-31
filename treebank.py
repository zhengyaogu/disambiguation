# -*- coding: utf-8 -*-

example = "( (SBARQ (WHNP (WDT What) (NNS debts)) (SQ (VBD did) (NP (NNP Qintex) (NN group)) (VP (VB leave))) (. ?)))"
       
class LabeledTree:
    def __init__(self, label, children):
        self.label = label
        self.children = children
        
    def __str__(self):
        if len(self.children) == 0:
            return self.label
        else:
            label = self.label
            if label == None:
                label = ''
            result = '(' + label
            for child in self.children:
                result += ' ' + str(child)
            result += ')'
            return result
        
    def leaves(self):
        if len(self.children) == 0:
            return [self.label]
        else:
            result = []
            for child in self.children:
                result += child.leaves()
            return result

    def spans(self):
        result, _ = self._spans_helper(0)
        return result
        
    def _spans_helper(self, offset=0):
        if len(self.children) == 0:
            return [], offset + 1
        result = [(offset, len(self.leaves()) + offset)]
        child_offset = offset
        for child in self.children:
            child_spans, child_offset = child._spans_helper(child_offset)            
            result += child_spans
        return result, child_offset
        
def all_spans(num_toks):
    spans = []
    for i in range(num_toks):
        for j in range(i+1, num_toks+1):
            spans.append((i,j))
    return spans

def tokenize_lisp(lisp_str):
    lisp_str = lisp_str.replace('(', ' ( ')
    lisp_str = lisp_str.replace(')', ' ) ')
    basic_toks = lisp_str.split()
    return basic_toks

def next_tree(toks):
    if len(toks) == 0:
        return None, toks
    elif toks[0] != '(' and toks[0] != ')':
        return [toks[0]], toks[1:]
    elif toks[0] == '(':
        openparens = 1
        index = 1
        while openparens > 0:
            if toks[index] == '(':
                openparens += 1
            elif toks[index] == ')':
                openparens -= 1
            index += 1
        return toks[:index], toks[index:]

def read_labeled_tree_helper(toks):
    if len(toks) == 0:
        return None
    elif toks[0] != '(' and toks[0] != ')':
        return LabeledTree(toks[0], [])
    elif toks[0] == '(' and toks[1] != '(':
        label = toks[1]
        rest = toks[2:]
    elif toks[0] == '(' and toks[1] == '(':
        label = None
        rest = toks[1:]
    children = []
    while rest[0] != ')':
        child, rest = next_tree(rest)
        children.append(read_labeled_tree_helper(child))
    return LabeledTree(label, children)
    
def read_labeled_tree(lisp_str):
    return read_labeled_tree_helper(tokenize_lisp(lisp_str))

QUESTION_BANK_PATH = '/Users/hopkinsm/Projects/data/parsing/treebanks/question-bank/stanford-patch-1.0/constituency'
from os.path import join

def read_question_bank(path = QUESTION_BANK_PATH):
    train_file = join(path, 'qbank.train.trees')
    dev_file = join(path, 'qbank.dev.trees')
    test_file = join(path, 'qbank.test.trees')
    with open(train_file, 'r') as inhandle:
        train = []
        for line in inhandle:
            line = line.strip()
            train.append(read_labeled_tree(line))
    with open(dev_file, 'r') as inhandle:
        dev = []
        for line in inhandle:
            line = line.strip()
            dev.append(read_labeled_tree(line))
    with open(test_file, 'r') as inhandle:
        test = []
        for line in inhandle:
            line = line.strip()
            test.append(read_labeled_tree(line))
    return train, dev, test

