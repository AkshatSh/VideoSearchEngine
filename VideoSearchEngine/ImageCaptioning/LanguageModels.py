'''
Using Obj2Text model as described here:

https://github.com/xuwangyin/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

'''

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torchvision.models
from torch.autograd import Variable

class EncoderCNN(nn.Module):
    '''
    Encoder CNN to extract all the features from an image

    Run some CNN for feature extraction (Paper uses ResNet-152)
    Pass results onto further encoder and decoder layers
    '''
    def __init__(self, embedded_size, bbox_model=None):
        super(EncoderCNN, self).__init__()

        # set up resnet to extract CNN features
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        self.linear = nn.Linear(resnet.fc.in_features, embedded_size)

        # normalize the batch after the embedding
        self.batch_norm = nn.BatchNorm1d(embedded_size, momentum=0.01)

        # intialize the weights with the distribution defined
        self.init_weights()
    
    def init_weights(self):
        '''
        Initialize the weights using a normal distribution
        Intialize bias to be 0
        '''
        self.linear.weight.data.normal_(0.0, 0.2)
        self.linear.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.resnet.forward(x)

        # convert results of bounding box detection to a variable
        x = Variable(x.data)

        # reduce to a single row for each batch
        x = x.view(x.size(0), -1)

        # apply a linear layer
        x = self.linear(x)

        # normalize the results
        x = self.bn(x)
        return x

class LayoutEncoder(nn.Module):
    '''
    Encode the layout using the bounding box
    image detection
    '''
    def __init__(self, layout_encoding_size, hidden_size, vocab_size, num_layers):
        super(LayoutEncoder, self).__init__()
        
        # create an embedding matrix for each of the labels in the 
        # encoder
        self.label_encoder = nn.Embedding(vocab_size, layout_encoding_size)

        # Use a linear layer to encode the location of each object
        self.location_encoder = nn.Linear(4, layout_encoding_size)

        # Use the LSTM to encode a squence of the layout_encodings
        self.lstm = nn.LSTM(layout_encoding_size, hidden_size, num_layers, batch_first=True)

        # intialize weights
        self.init_weights()
    
    def init_weights(self):
        self.label_encoder.weight.data.uniform_(-0.1, 0.1)
        self.location_encoder.weight.data.uniform_(-0.1, 0.1)
        self.location_encoder.bias.data.fill_(0)
    
    def forward(self, label_seqs, location_seqs, lengths):
        # sort sequences acording to length in the batch dimension
        batch_idx = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse=True)
        reverse_batch_idx = torch.LongTensor([batch_idx.index(i) for i in range(len(batch_idx))])

        # sort the lengths
        lens_sorted = sorted(lengths, reverse=True)

        # sort the sequences as well
        label_seqs_sorted = torch.index_select(label_seqs, 0, torch.LongTensor(batch_idx))
        location_seqs_sorted = torch.index_select(location_seqs, 0, torch.LongTensor(batch_idx))


        # apply CUDA when available
        if torch.cuda.is_available():
            reverse_batch_idx = reverse_batch_idx.cuda()
            label_seqs_sorted = label_seqs_sorted.cuda()
            location_seqs_sorted = location_seqs_sorted.cuda()

        label_seqs_sorted_var = Variable(label_seqs_sorted, requires_grad=False)
        location_seqs_sorted_var = Variable(location_seqs, requires_grad=False)

        # encode the labels
        label_encoding = self.label_encoder(label_seqs_sorted_var)

        # encode the location sequence

        # flatten the location sequences to rows of 4
        location_seqs_sorted_var = location_seqs_sorted_var.view(-1, 4)

        # use the location encoding linear layer to encode the location
        location_encoding = self.location_encoder(location_seqs_sorted_var)

        # create the encoding to be [batch_size, ____, layout_encoding]
        location_encoding = location_encoding.view(label_encoding.size(0), -1, location_encoding.size(1))

        # [batch_size, max_seq_len, layout_encoding_size]
        layout_encoding = label_encoding + location_encoding

        # pack everything for the LSTM layer
        packed = pack(layout_encoding, lens_sorted, batch_first=True)
        hiddens, _ = self.lstm(packed)

        # unpack the hidden layers from the lstm output

        # use the output of the LSTM
        # [batch_size, max_seq_len, layout_encoding_size]
        hiddens_unpacked = unpack(hiddens, batch_first=True)[0]

        # Create a new tensor of size [batch_size, 1, layout_encoding_size]
        # and zero out the entreis
        last_hidden_idx = torch.zeros(hiddens_unpacked.size(0), 1, hiddens.unpacked.size(2)).long()

        # TODO: improve this with vectorize or some shit (remove the loop)
        for i in range(hiddens_unpacked.size(0)):

            # set the 2nd index to the length of the sequence
            last_hidden_idx[i, 0, :] = lens_sorted[i] - 1
        
        if torch.cuda.is_available():
            last_hidden_idx = last_hidden_idx.cuda()
        
        # replace the first index with the last_hidden_idx of the hiddens_unpacked
        last_hidden = torch.gather(hiddens_unpacked, 1, Variable(last_hidden_idx, requires_grad=False))
        last_hidden = torch.index_select(last_hidden, 0, Variable(reverse_batch_idx, requires_grad=False))

        # convert back to original batch order
        last_hidden = torch.index_select(last_hidde, 0, Variable(reverse_batch_idx, requires_grad=False))

        return last_hidden

class DecoderRNN(nn.Module):
    '''
    Decode the encoder output and convert it 
    to a format ready to create the output sequence
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        # conver the vocab to the embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # send through the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        # convert output to words
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    def forward(self, features, captions, lengths):
        '''
        Decode Encoder output and produce image captions
        '''
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        '''
        Sample Captions for a given image
        '''
        sampled_ids = []

        # add a dimension at the first index to be [_, 1, _, _]
        inputs = features.unsqueeze(1)
        for _ in range(20): # 20 is max sample length
            hiddens, states = self.lstm(inputs, states) # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            predicted = outputs.max(1)[1] # gready search
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)
        
        sampled_ids = torch.cat(sampled_ids, 1) # [batch_size, 20]
        return sampled_ids.unsqueeze()