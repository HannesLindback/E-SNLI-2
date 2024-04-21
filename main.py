import csv
import argparse
import pandas as pd
from model import EncoderDecoder, PredictLabel
from utils import Load, Data, Encode, ToTensor, Word_Tokenizer
from mapping import VocabMapping
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt


class PredictExplanations:
    
    def __init__(self, args):
        self.args            = args

        self.vocab_size      = None
        self.pad_index       = None
        self.sos_index       = None
        self.eos_index       = None
        self.mapping         = None

        self.model           = None
        self.classifier      = None
        self.criterion       = self._criterion(args)
        self.optimizer       = self._optimizer(args)
        self.lr_scheduler    = self._lr_scheduler(args)

        self.current_loss    = None
        self.training_losses = []
        self.valid_losses    = []
        self.epoch_loss      = []

        self.highlighted     = None

        if args.fp16:
            self.scaler      = torch.cuda.amp.GradScaler()

    def generate(self, test_data, path='generated.csv', epoch=None):
        """Generates labels and explanations."""

        test_dataloader = DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=False)

        self.model.eval()

        correct_labels = []
        all_labels = []
        sent_1_highlight = list(self.highlighted['Sentence1_marked_1'])
        sent_2_highlight = list(self.highlighted['Sentence2_marked_1'])
        i = 0

        with open(str(epoch)+self.args.experiment+path, 'a', encoding='utf-8') as fhand:
            writer = csv.writer(fhand, delimiter='\t')
            writer.writerow(['Predicted', 'Correct', 'Marked'])
            for batch in tqdm(test_dataloader, desc='Predicting', total=len(test_data), unit='sentence'):
                source, gold = batch['x'], batch['y']
                
                if self.args.experiment == 'explanation':
                    gold_expl = gold
                    u, v = source
                    u_mask = (u != self.pad_index)
                    v_mask = (v != self.pad_index)
                    y_expl = self.model.greedy_decode(u, v, u_mask, v_mask, self.sos_index, self.eos_index)
                elif self.args.experiment == 'label':
                    gold_label = gold
                    y_label = self.model.greedy_decode(source, None)
                    
                if  self.args.experiment == 'explanation':
                    predicted_expl = self.mapping.decode(y_expl.tolist()[0])
                    correct_expl = self.mapping.decode(gold_expl.tolist()[0])
                    
                    marked_words_1 = [word for word in sent_1_highlight[i].split()
                                    if word[0] == '*']
                    marked_words_2 = [word for word in sent_2_highlight[i].split()
                                    if word[0] == '*']
                    marked_words = ' '.join(marked_words_1 + marked_words_2)
                    
                    writer.writerow([predicted_expl, correct_expl, marked_words])
                
                if self.args.experiment == 'label':
                    if gold_label.item() == y_label.item():
                        correct_labels.append(y_label.item())
                    all_labels.append((y_label.item(), gold_label.item()))
                
                i += 1
        
        if self.args.experiment == 'label':
            with open(str(epoch)+self.args.experiment+'label_results.txt', 'w', encoding='utf-8') as fhand:
                fhand.write(f'Label accuracy: {len(correct_labels) / len(test_data)}\n')
                fhand.write('\nPredicted label,Correct label\n')
                for pred, gold in all_labels:
                    fhand.write(f'{pred},{gold}\n')
                
    def train(self, train_data, dev_data, test_data=None,
              loss=None, start_epoch=None):
        """Trains model"""
        
        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=self.args.batch_size,
                                      shuffle=True)
  
        dev_dataloader = DataLoader(dataset=dev_data,
                                    batch_size=self.args.batch_size,
                                    shuffle=True)
        
        best_loss = loss if loss else float('inf')
        start = start_epoch if start_epoch else 0
        
        for epoch in range(start, self.args.epochs):
            ##################### TRAIN #####################
            self.model.train()
            average_train_loss = self._single_epoch_train(
                                                train_dataloader,
                                                len(train_data),
                                                desc=f' Training epoch {epoch+1}',
                                                train=True
                                                )
            print(f'Average train loss for epoch {epoch+1}: {average_train_loss}')
            
            if (epoch+1) % 5 == 0 and epoch != 0:
                self.lr_scheduler.step()
                print(f'Current learning rate: {self.lr_scheduler.get_last_lr()}')
            
            self.generate(test_data, epoch=epoch+1)
            
            self._plot(plot_loss_for_epoch=True)
            
            with open('log.log', 'a', encoding='utf-8') as fhand:
                fhand.write(f'Average train loss for epoch {epoch+1}: {average_train_loss}\n')
                
            ##################### VALIDATE #####################
            self.model.eval()
            with torch.no_grad():
                average_dev_loss = self._single_epoch_train(
                                            dev_dataloader,
                                            len(dev_data),
                                            desc=f'Validating epoch {epoch+1}',
                                            train=False
                                            )
            print(f'Average valid loss for epoch {epoch+1}: {average_dev_loss}')
            
            with open('log.log', 'a', encoding='utf-8') as fhand:
                fhand.write(f'Average valid loss for epoch {epoch+1}: {average_dev_loss}\n')
            
            if average_dev_loss < best_loss:
                self.save_model(epoch=epoch+1,
                                loss=self.current_loss,
                                path=self.args.save_checkpoint)
            
        self._plot()
    
    def _plot(self, plot_loss_for_epoch=False):
        """Plot losses!"""
        
        if plot_loss_for_epoch:
            plt.figure()
            plt.plot([step for step in range(len(self.epoch_loss))],
                    self.epoch_loss, label='Training loss per step')
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(fname='step_loss_curve.png')
            plt.close()
        else:
            plt.figure()
            plt.plot([epoch for epoch in range(len(self.training_losses))],
                    self.training_losses, label='Training loss per epoch')
            plt.plot([epoch for epoch in range(len(self.training_losses))],
                    self.valid_losses, label='Validation loss per epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(fname='loss_curve.png')
            plt.close()
    
    ##################### INITIALIZERS #####################
    def _optimizer(self, args):
        if args.optimizer == 'adamw':
            optim = torch.optim.AdamW
        elif args.optimizer == 'adam':
            optim = torch.optim.Adam
        elif args.optimizer == 'adagrad':
            optim = torch.optim.Adagrad
        elif args.optimizer == 'adadelta':
            optim = torch.optim.Adadelta
        return optim
    
    def _criterion(self, args):
        if args.loss_function == 'nll':
            loss = nn.NLLLoss
        elif args.loss_function == 'cross_entropy':
            loss = nn.CrossEntropyLoss
        return loss
    
    def _lr_scheduler(self, args):
        if args.lr_scheduler == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR
        return lr_scheduler
    
    
    ##################### TRAINING #####################
    def _forward(self, batch):
        """Perform the forward pass"""
        
        source, target = batch['x'], batch['y']
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            if self.args.experiment == 'explanation':
                u, v = source
                u_mask = (u != self.pad_index)
                v_mask = (v != self.pad_index)
                target_in, target_out = target[:, :-1], target[:, 1:]
                output = self.model(u, v, target_in, u_mask, v_mask)
                loss = self.criterion(output.permute(0,2,1), target_out)
            elif self.args.experiment == 'label':
                source_mask = (source != self.pad_index)
                output = self.model(source, source_mask)
                loss = self.criterion(output, target)
                
        return loss
    
    def _single_epoch_train(self, dataloader, data_size, desc, train=False):
        """Performs one iteration of training through the dataset."""
        
        accumulated_loss = 0.0
        
        with tqdm(dataloader,
                    total=data_size//self.args.batch_size+1,
                    unit='batch',
                    desc=desc) as batches:
            for batch in batches:
                loss = self._forward(batch)
                loss_item = loss.item()
                accumulated_loss += loss_item
                batches.set_postfix(loss=loss_item)
                
                if train:
                    self.epoch_loss.append(loss_item)
                    self.current_loss = loss
                    self._backward(loss)
                
        average_loss = accumulated_loss / (data_size//self.args.batch_size+1)
        
        if train:
            self.training_losses.append(average_loss)
        else:
            self.valid_losses.append(average_loss)
                
        return average_loss
    
    def _backward(self, loss):
        """Perform the backward pass"""
        
        self.model.zero_grad()
        if self.args.fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.optimizer.step()
    
    
    ##################### SAVE/LOAD MODEL #####################
    def save_model(self, epoch, loss, path):
        """Save model to pytorch checkpoint."""
        
        print('Saving checkpoint')
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss                    
                   }, str(epoch)+path)
        
    def load_model(self, path):
        """Load model from pytorch checkpoint"""
        
        print('Loading model from checkpoint')
        
        self.model.to(self.args.device)
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        self.current_loss = checkpoint['loss']
        return start_epoch, self.current_loss
    
    
    ##################### PREPROCESSING #####################
    def preprocess(self):
        """Load datasets and create mapping of all word types in dataset."""
        
        train_data, mapping = self._get_dataset(self.args.train_path)
        dev_data, _ = self._get_dataset(self.args.dev_path, mapping)
        test_data, _ = self._get_dataset(self.args.test_path, mapping, test=True)
        self.pad_index = mapping.pad_id
        self.sos_index = mapping.sos_id
        self.eos_index = mapping.eos_id
        self.vocab_size = len(mapping)
        self.mapping = mapping
        return train_data, dev_data, test_data
    
    def _get_dataset(self, path, mapping=None, test=False):
        content = Load(path)
        data = content.data
        labels = content.labels
        self.highlighted = content.highlighted
        
        if self.args.experiment == 'explanation':
            source = [(premise, hypothesis) for premise, hypothesis in
                      zip(list(data['Sentence1']), list(data['Sentence2']))]
            target = [expl for expl in list(data['Explanation_1'])]
            SOS_EOS = False
        elif self.args.experiment == 'label':
            source = [expl for expl in list(data['Explanation_1'])]
            target = list(labels)
            SOS_EOS = True
            
        if mapping is None:
            if self.args.tokenization_type == 'word':
                data = self._tokenize_data(data)
            mapping = VocabMapping(data, tok_type=self.args.tokenization_type,
                                   SOS=SOS_EOS)
            self.pad_index = mapping.pad_id
            
        transform = [Encode(mapping=mapping,
                            max_seq_len=self.args.padding_size,
                            pad_id=mapping.pad_id,
                            SOS_EOS=SOS_EOS),
                    ToTensor()]
        
        dataset = Data(x=source,
                    y=target,
                    transform=transforms.Compose(transform))
        
        return dataset, mapping
    
    def _tokenize_data(self, dataframe):
        """Tokenize a dataset."""
        
        tokenize = Word_Tokenizer()
        tok_columns = []
        for _, column in dataframe.items():
            sents = []
            for sentence in tqdm(column, desc=f'Tokenizing sentences'):
                sents.append(tokenize(sentence.lower()))
            tok_columns.append(sents)
        
        data = pd.DataFrame({name: rows for name, rows in 
                             zip(dataframe.columns, tok_columns)})
        return data
    

def optimizer_to(optim, device):
    """Stupid helper function for stupid pytorch"""
    
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment',
                    help='Which e-SNLI experiment to test.\
                     Can be explanation or label')
    ap.add_argument('--train_path')
    ap.add_argument('--dev_path')
    ap.add_argument('--test_path')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save_checkpoint', default='model.pt')
    ap.add_argument('--load_checkpoint')
    ap.add_argument('--train', default=True, type=bool)
    ap.add_argument('--test', default=True, type=bool)

    ap.add_argument('--tokenization_type', default='word')
    ap.add_argument('--max_seq_len', default=256, type=int,
                    help='Max length of sequences passed to model')
    ap.add_argument('--padding_size', type=int, default=256)
    ap.add_argument('--batch_size', default=128, type=int)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--loss_function', default='cross_entropy')
    ap.add_argument('--optimizer', default='adam')
    ap.add_argument('--lr_scheduler', default='exponential')
    ap.add_argument('--fp16', default=True, type=bool)
    ap.add_argument('--alpha', type=int, default=0.6)

    ap.add_argument('--d_model', type=int, default=512)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--num_encoder_layers', type=int, default=6)
    ap.add_argument('--num_decoder_layers', type=int, default=6)
    ap.add_argument('--dim_feedforward', type=int, default=1024)
    ap.add_argument('--dropout', type=int, default=0.5)
    ap.add_argument('--lr', type=int, default=0.0001)
    ap.add_argument('--weight_decay', type=int, default=5e-5)

    args = ap.parse_args()
    print(f'Running experiment {args.experiment}')
    p_e = PredictExplanations(args)

    train_data, dev_data, test_data = p_e.preprocess()

    if args.experiment == 'explanation':
        p_e.model = EncoderDecoder(
                                src_vocab_size=p_e.vocab_size,
                                tgt_vocab_size=p_e.vocab_size,
                                d_model=args.d_model,
                                n_heads=args.nhead,
                                n_enc_layers=args.num_encoder_layers,
                                n_dec_layers=args.num_decoder_layers,
                                dim_ff=args.dim_feedforward,
                                dropout=args.dropout,
                                max_seq_len=args.max_seq_len,
                                n_labels=3
                                )
    elif args.experiment == 'label':
        p_e.model = PredictLabel(vocab_size=p_e.vocab_size,
                                 max_seq_len=args.max_seq_len,
                                 d_model=args.d_model,
                                 n_heads=args.nhead,
                                 n_enc_layers=args.num_encoder_layers,
                                 dim_ff=args.dim_feedforward,
                                 dropout=args.dropout,
                                 n_labels=3)

    p_e.optimizer = p_e.optimizer(p_e.model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
    
    p_e.criterion = p_e.criterion(ignore_index=p_e.pad_index)
    p_e.lr_scheduler = p_e.lr_scheduler(p_e.optimizer, gamma=0.1, verbose=True)

    if not args.load_checkpoint:
        epoch, loss = None, None
    elif args.load_checkpoint:
        epoch, loss = p_e.load_model(args.load_checkpoint, train_data)

    p_e.model.to(args.device)
    optimizer_to(p_e.optimizer, args.device)

    if args.train:
        p_e.train(train_data, dev_data, test_data=test_data, loss=loss, start_epoch=epoch)
    if args.test:
        p_e.generate(test_data, epoch='Finished')

if __name__ == '__main__':
    
    main()