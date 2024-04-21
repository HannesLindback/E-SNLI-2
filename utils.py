import torch
from torch.utils.data import Dataset
import string
import pandas as pd
import numpy as np


class Load:
    """Loads the data to a pandas dataframe."""
    
    def __init__(self, dataset_path):
        
        self.dataset_path = dataset_path
        
        df = pd.read_csv(dataset_path)
        
        # Remove rows with missing columns.
        df.dropna(inplace=True)
        
        self.data = pd.concat([df['Sentence1'],
                               df['Sentence2'],
                               df['Explanation_1']],
                              axis=1)
        self.labels = df['gold_label']
        self.highlighted = pd.concat([df['Sentence1_marked_1'], df['Sentence2_marked_1']], axis=1)

class Word_Tokenizer:
    """A pretty simple tokenizer for tokenizing sequences on a word-level.
    
    Faster than NLTK's tokenizer but not as fast as the Huggingface tokenizers."""
    
    def __init__(self):
        self.special_tokens = self._special_tokens()
        self.punct = set(string.punctuation)
        self.sentence = None
        
    def _special_tokens(self):
        special = {'u.k.', 'u.s.a.', '...', "don't", "can't", "won't",
                   "how's", "how're", "where's", "i'm", "you're", "he's",
                   "she's", "isn't", "aren't", "we're", "they're", "wasn't"}
        return special

    def _split(self, unsplit_tokens):
        tokens = []
        text = []
        for c in unsplit_tokens:
            if c in self.punct:
                if len(text) > 0:
                    tokens.append(''.join(text))
                    text = []
                tokens.append(c)
            else:
                text.append(c)
        
        return tokens
        
    def _tokenize(self, word):
        word = ''.join(word)
        
        if word in self.special_tokens:
            tokenized = word
        elif word[0] in self.punct or word[-1] in self.punct:
            tokenized = self._split(word)
        else:
            tokenized = word
        return tokenized
    
    def _add(self, word, tokens):
        tokenized = self._tokenize(word)
        if isinstance(tokenized, list):
            tokens.extend(tokenized)
        else:
            tokens.append(tokenized)
        return tokens
            
    def __call__(self, sentence):
        self.sentence = sentence
        
        tokens = []
        word = []
        for c in sentence.strip():
            if c == ' ' and len(word) > 0:
                tokens = self._add(word, tokens)
                word = []
            if c != ' ':
                word.append(c)
        
        tokens = self._add(word, tokens)
        
        return tokens


class Data(Dataset):
    
    def __init__(self, x, y, transform=None) -> None:
        self.x, self.y = x, y
        self.transform  = transform
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'x': self.x[idx], 'y': self.y[idx]}
            
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
class Encode:
    """Helper class for encoding tokens to indices."""
    
    def __init__(self, mapping, max_seq_len, pad_id=-1, SOS_EOS=False):
        self.mapping = mapping
        self.pad_size = max_seq_len
        self.pad_id = pad_id
        self.SOS_EOS = SOS_EOS
        
    def __call__(self, sample):
        source_raw, target_raw = sample['x'], sample['y']
    
        source = self._encode(source_raw)
        target = self._encode(target_raw)

        return {'x': source, 'y': target}
    
    def _encode(self, raw):
        above_max_seq_len_error_msg = \
        'Sequence length cannot exceed maximum sequence length!'
        
        if self.mapping.labels.get(raw, None):
            raw = self.mapping.labels.get(raw, raw)
            encoded = self.mapping._token2index[raw]
            
        else:
            ids = self.mapping[raw]
            
            try:
                assert len(ids) <= self.pad_size, above_max_seq_len_error_msg
            except AssertionError:
                print(self.mapping.decode(ids))
                raise AssertionError
            
            encoded = np.array([self.pad_id]*(self.pad_size))
            encoded[:len(ids)] = ids

        return encoded
    

class ToTensor:
    """Converts source and target numpy arrays to tensors."""

    def __call__(self, sample):
        source, target = sample['x'], sample['y']
        
        source = self._totensor(source)
        target = self._totensor(target)
        
        return {'x': source, 'y': target}

    def _totensor(self, ids):
        tens = torch.tensor(ids, dtype=torch.long)
        
        if torch.cuda.is_available():
            return tens.cuda()
        else:
            return tens