from collections import Counter
import pathlib
import os.path
import pandas as pd
import sentencepiece as spm
import string
from utils import Word_Tokenizer


class VocabMapping:
    
    def __init__(self, dataset,
                 tok_type='word',
                 model_prefix='spm',
                 sp_data_file='data.txt',
                 sp_vocab_size=5000,
                 SOS=False):
        """Creates a token to index mapping.
        
        Args:
            dataset: A Pandas object with one or more columns with
                     tokenized data, or if sentencepiece tokenization,
                     a Pandas object with raw data.
            tok_type: A string specifying if the tokenization type is
                      word level or Sentencepiece.
            model_prefix: The prefix to the Sentencepiece model name.
            sp_data_file: The path to the temporary data file that SP uses."""
        
        print('Creating mapping...')
        
        self.tok_type = tok_type
        self.tokenize = Word_Tokenizer()
        self._check_for_errors(dataset, tok_type)
        self.sp_vocab_size = sp_vocab_size
        self.labels = {'neutral': 'NEU',
                       'entailment': 'ENT',
                       'contradiction': 'CON'}
        self.SOS = SOS
        
        if tok_type == 'word':
            idx2token = ['ENT'] + \
                        ['NEU'] + \
                        ['CON'] + \
                        ['<SOS>'] + \
                        ['<EOS>'] + \
                        ['<PAD>'] + \
                        ['<UNK>'] + \
                        list(self._get_n_word_types(dataset))
                        
            self._token2index = {tok:idx for idx, tok in enumerate(idx2token)}
            self._idx2token = {idx:tok for idx, tok in enumerate(idx2token)}
            self.pad_id = self._token2index['<PAD>']
            self.sos_id = self._token2index['<SOS>']
            self.eos_id = self._token2index['<EOS>']
            
        elif tok_type == 'sentencepiece':
            if not os.path.exists(f'{model_prefix}.model'):
                data = pd.concat(dataset, axis=0)
                data.to_csv(sp_data_file, index=False)
                spm.SentencePieceTrainer.train(
                    f'--input={sp_data_file} \
                    --model_prefix={model_prefix} \
                    --vocab_size={str(self.sp_vocab_size)} \
                    --shuffle_input_sentence=true'
                    )
                
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(f'{model_prefix}.model')
            self.pad_id = self.sp.pad_id()
            
            if os.path.exists(sp_data_file):
                pathlib.Path(sp_data_file).unlink()
    
    def __getitem__(self, raw_string):
        raw_string = raw_string.lower()
        
        if self.tok_type == 'word':
            tokens = self.tokenize(raw_string)
            
            ids = [self._token2index['<SOS>']]
            for token in tokens:
                try:
                    ids.append(self._token2index[token])
                except KeyError:
                    ids.append(self._token2index['<UNK>'])
            ids.append(self._token2index['<EOS>'])
            
        elif self.tok_type == 'sentencepiece':
            ids = [self.sp.bos_id()]
            ids.extend(self.sp.encode_as_ids(raw_string))
            ids.append(self.sp.eos_id())
        return ids
    
    def __len__(self):
        if self.tok_type == 'word':
            length = len(self._token2index)
        elif self.tok_type == 'sentencepiece':
            length = self.sp_vocab_size
        return length
    
    def decode(self, encoded_seq):
        """Decodes a sequence of token-indices to string."""
        
        def detokenize(seq):
            punct = set(string.punctuation)
            detokenized_seq = [seq[0]]
            for token in seq[1:]:
                if token in punct:
                    detokenized_seq[-1] = ''.join([detokenized_seq[-1], token])
                else:
                    detokenized_seq.append(token)
            return ' '.join(detokenized_seq)
        
        if self.tok_type == 'word':
            decoded_seq = [self._idx2token[token]
                           for token in encoded_seq if token != self.pad_id]
            decoded_seq = detokenize(decoded_seq)
            
        elif self.tok_type == 'sentencepiece':
            decoded_seq = self.sp.decode(encoded_seq)
            
        return decoded_seq
    
    def _check_for_errors(self, dataset, tok_type):
        error_msg_dataset_type = "Argument dataset should \
                                  be a pd.Dataframe object!"
        error_msg_tok_type = "Argument dataset should be the string 'word' or \
                             the string 'sentencepiece'!"
        assert isinstance(dataset, pd.DataFrame), error_msg_dataset_type
        assert tok_type.lower() == 'word' or \
               tok_type.lower() == 'sentencepiece', error_msg_tok_type
                             
    def _get_n_word_types(self, tok_columns):
        tokens = [token for _, column in tok_columns.items()
                        for sentence in column
                        for token in sentence]
        return {w for w, c in Counter(tokens).items() if c > 2}