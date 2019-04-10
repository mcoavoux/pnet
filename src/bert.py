import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from pytorch_pretrained_bert import BertTokenizer, BertModel


BERT_DIM = 768


def get_fine_tuned_bert(modelname):
    logging.info("Loading fine-tuned BERT...")
    bert = torch.load("{}/model_fine_tuned_bert".format(modelname))
    return bert


## The following class is taken from: 
## https://github.com/hassyGo/pytorch-playground/blob/master/gradient_reversal/gradient_reversal.py
class GradientReversal(torch.autograd.Function):

    def __init__(self, scale_):
        super(GradientReversal, self).__init__()

        self.scale = scale_

    def forward(self, inp):
        return inp.clone()

    def backward(self, grad_out):
        return -self.scale * grad_out.clone()

class BertEncoder(nn.Module):

    def __init__(self):
        super(BertEncoder, self).__init__()
#        bert-base-uncased: 12-layer, 768-hidden, 12-heads, 110M parameters
#        bert-large-uncased: 24-layer, 1024-hidden, 16-heads, 340M parameters
#        bert-base-cased: 12-layer, 768-hidden, 12-heads , 110M parameters
#        bert-large-cased: 2
        bert_id = "bert-base-cased"
        self.tokenizer = BertTokenizer.from_pretrained(bert_id, do_lower_case=False)
        self.bert = BertModel.from_pretrained('bert-base-cased')

        # just to get the device easily
        self.device = nn.Parameter(torch.zeros(1))

        self.replace = {"-LRB-": "(", "-RRB-": ")"}

        self.dim = BERT_DIM


    def forward(self, sentence, batch = False):
        """
        input:
            sentence: list[str] if not batch else list[list[str]]
            batch: bool
        output:
            list[Tensor] if not batch else list[list[Tensor]]
            returns one contextual vector per word in each sentence
        """
        if batch:
            batch_tokens = []
            batch_masks = []
            batch_lengths = []
            for token_list in sentence:
                token_list = token_list[:500]
                for i in range(len(token_list)):
                    if token_list[i] in self.replace:
                        token_list[i] = self.replace[token_list[i]]

                    wptokens = []
                    for token in token_list:
                        wp = self.tokenizer.wordpiece_tokenizer.tokenize(token)
                        wptokens.extend(wp)
                    wptokens = wptokens[:500]

                    mask = [0 if tok[:2] == "##" else 1 for tok in wptokens]
                    mask = torch.tensor(mask, dtype=torch.uint8, device=self.device.device)

                    indexed_tokens = self.tokenizer.convert_tokens_to_ids(wptokens)
                    tokens_tensor = torch.tensor([indexed_tokens], device = self.device.device)

                batch_tokens.append(tokens_tensor.view(-1))
                batch_masks.append(mask)
                batch_lengths.append(len(wptokens))
            
            padded = pad_sequence(batch_tokens, batch_first=True)
            mask = padded != 0
            encoded_layers,_ = self.bert(input_ids=padded, attention_mask=mask, output_all_encoded_layers=False)
            split_layers = encoded_layers.split([1 for _ in sentence])
            assert(len(split_layers) == len(batch_masks))
            filtered_layers = [layer.squeeze(0)[:l][m] for layer, l, m in zip(split_layers, batch_lengths, batch_masks)]

            return filtered_layers
        else:
            for i in range(len(sentence)):
                if sentence[i] in self.replace:
                    sentence[i] = self.replace[sentence[i]]

            wptokens = []
            for token in sentence:
                wp = self.tokenizer.wordpiece_tokenizer.tokenize(token)
                wptokens.extend(wp)
            wptokens = wptokens[:500]

            mask = [0 if tok[:2] == "##" else 1 for tok in wptokens]
            mask = torch.tensor(mask, dtype=torch.uint8, device=self.device.device)

            indexed_tokens = self.tokenizer.convert_tokens_to_ids(wptokens)
            tokens_tensor = torch.tensor([indexed_tokens], device = self.device.device)

            encoded_layers,_ = self.bert(tokens_tensor, output_all_encoded_layers=False)
            filtered_layers = encoded_layers.squeeze(0)[mask]
            return filtered_layers



if __name__ == "__main__":
    bert = BertEncoder()

    bert.cuda()

    sentence = "The bill , whose backers include Chairman Dan Rostenkowski -LRB- D. , Ill. -RRB- , would prevent the Resolution Trust Corp. from raising temporary working capital by having an RTC-owned bank or thrift issue debt that would n't be counted on the federal budget .".split()

    print(len(sentence))
    output = bert(sentence)
    print([i.shape for i in output])
    print(len(output))

    output = bert([sentence, sentence], batch = True)
    print([len(i) for i in output])






