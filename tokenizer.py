# -*- coding: utf-8 -*-
from pytorch_transformers import BertTokenizer, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def dual_tokenize(parse):
    def align_next_parse_token(parse_tokens, bert_tokens):
        next_parse_token = parse_tokens[0]
        matcher = ''
        bert_token_index = 0
        while len(matcher) < len(next_parse_token):
            next_bert_token = bert_tokens[bert_token_index]
            if next_bert_token.startswith("##"):
                matcher += next_bert_token[2:]
            else:
                matcher += next_bert_token
            bert_token_index += 1
        return bert_token_index

    def map_span(span, token_ends):
        (start, stop) = span
        prev = 0
        if start >= 1:
            prev = token_ends[start-1]
        return (prev, token_ends[stop-1])

    parse_tokens = parse.leaves()
    parse_token_str = ' '.join(parse_tokens)
    bert_tokens = tokenizer.tokenize(parse_token_str)
    bert_token_index = 0
    alignment = []
    for i in range(len(parse_tokens)):
        bert_token_index += align_next_parse_token(parse_tokens[i:], 
                                                   bert_tokens[bert_token_index:])
        alignment.append(bert_token_index)        
    return parse_tokens, bert_tokens, lambda span: map_span(span, alignment)

def bert_tokens_and_spans(parse):
    parse_spans = parse.spans()
    no_singletons = [(start, stop) for (start, stop) in parse_spans if stop-start >=2]
    _, bert_toks, span_map = dual_tokenize(parse)
    bert_spans = [span_map(span) for span in no_singletons]
    return bert_toks, set(bert_spans)

def print_spans(toks, spans):
    for (start, stop) in spans:
        print(' '.join(toks[start:stop]))
    