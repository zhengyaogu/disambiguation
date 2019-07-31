import torch

from treebank import all_spans
from tokenizer import bert_tokens_and_spans
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
import pandas as pd 

config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel(config)    

def vectorize_instances(tokens, spans):
    """
    Converts a set of BERT tokens and spans into a tensor.
    
    """
    input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens)).unsqueeze(0)  # Batch size 1    
    outputs = bert(input_ids)[0].squeeze(0)
    encodings = [torch.cat([outputs[start], outputs[stop-1]]) for (start, stop) in spans]    
    return torch.stack(encodings)


def process_treebank(treebank, csvfile):
    """
    Converts the spans of the treebank into vectors and writes them to
    a CSV file.
    
    """
    with open(csvfile, 'w') as outhandle:
        for i, parse in enumerate(treebank):
            print(i)
            tokens, spans = bert_tokens_and_spans(parse)
            vec = vectorize_instances(tokens, spans)
            labeled_vec = torch.cat([torch.ones(vec.shape[0], 1), vec], dim=1)
            df = pd.DataFrame(labeled_vec.detach().numpy())
            outhandle.write(df.to_csv(header=False, index=False))    
            nontriv_spans = [(i,j) for (i,j) in all_spans(len(tokens)) if j-i >= 2]
            vec = vectorize_instances(tokens, list(set(nontriv_spans) - set(spans)))
            labeled_vec = torch.cat([torch.zeros(vec.shape[0], 1), vec], dim=1)
            df = pd.DataFrame(labeled_vec.detach().numpy())
            outhandle.write(df.to_csv(header=False, index=False))
            

