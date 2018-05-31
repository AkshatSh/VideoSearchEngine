import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


class YoloEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, bbox_model, bbox_encoding_size, vocab_size, vocab):
        super(YoloEncoder, self).__init__()
        num_layers = 2
        self.bbox_model = bbox_model

        # create an embedding matrix for the labels
        self.label_embedding = nn.Embedding(vocab_size, bbox_encoding_size)

        # takes the 4 points of the bounding box and encodes it to a custom
        # size
        self.bbox_encoder = nn.Linear(4, bbox_encoding_size)
        self.lstm = nn.LSTM(bbox_encoding_size, hidden_size, num_layers, batch_first=True)
        
        # vocab for encoding
        self.vocab = vocab
    
    def init_weights(self):
        self.label_embedding.weight.data.uniform_(-0.1, 0.1)
        self.bbox_encoder.weight.data.uniform_(-0.1, 0.1)
        self.bbox_encoder.bias.data.fill_(0)
    
    def forward(self, image):
        image = image.cpu().data.numpy()
        labels, bboxes, lengths = self.bbox_model.get_bbox(image)
        labels_one_hot = [torch.Tensor([self.vocab(token) for token in labels_n]) for labels_n in labels]

        # sort labels_one_hot and sort bboxes
        labels_target = torch.zeros(len(labels_one_hot), max(lengths), dtype=torch.long)
        for i, label_one_hot in enumerate(labels_one_hot):
            seq_end = lengths[i]
            labels_target[i, :seq_end] = label_one_hot[:seq_end]
        
        print(labels_target.shape)


        # sort the bboxes
        bboxes_target = torch.zeros(len(bboxes), max(lengths), 4)
        print(bboxes_target.shape)
        for i, bbox_seq in enumerate(bboxes):
            for j in range(len(bbox_seq)):
                bboxes_target[i, j, :] = bbox_seq[j]

        lengths = torch.Tensor(lengths)
        return self.forward_internal(labels_target, bboxes_target, lengths)
    
    def forward_internal(self, labels, bboxes, lengths):

        # sort the batches so that the longest sequence is the start
        # and the shortest is at the end
        # only sorts the indexes
        print("starting RNN encoding of bbox models")
        batch_idx_sorted = sorted(
            range(len(lengths)), 
            key=lambda k: lengths[k], 
            reverse=True)
        
        # uses the sorted batches and maps the locatoin of each index to the original position
        # so can be converted back to the normal order
        reverse_batch_idx = torch.LongTensor([batch_idx_sorted.index(i) for i in range(len(batch_idx_sorted))])

        # actually sort the lengths
        sorted_lengths = sorted(lengths, reverse=True)
        print(sorted_lengths)

        # lets sort the labels and bboxes as well according to this
        sorted_bboxes = torch.index_select(bboxes, 0, torch.LongTensor(batch_idx_sorted))
        sorted_labels = torch.index_select(labels, 0, torch.LongTensor(batch_idx_sorted))

        if torch.cuda.is_available():
            sorted_labels = sorted_labels.cuda()
            sorted_bboxes = sorted_bboxes.cuda()
            reverse_batch_idx = reverse_batch_idx.cuda()
        
        # create variable representations
        # non trainable because these are inputs
        labels_var = Variable(sorted_labels, requires_grad=False)
        bboxes_var = Variable(sorted_bboxes, requires_grad=False)

        # ENCODINGS

        # label encoding
        label_encoding = self.label_embedding(labels_var)

        # bboxex encoding

        # remove concept of batches and start just encoding it

        bboxes_var = bboxes_var.view(-1, 4)
        bboxes_encoding = self.bbox_encoder(bboxes_var)

        # restore the batch index
        bboxes_encoding = bboxes_encoding.view(label_encoding.size(0), -1, bboxes_encoding.size(1))

        # create final encoding

        # by SUMMING EACH THING ELEMENT WISE
        encoding = bboxes_encoding + label_encoding

        packed = pack(encoding, sorted_lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)

        # Grab the output from the hiddens once unpacked
        hiddens_unpacked = unpack(hiddens, batch_first=True)[0]

        last_hidden_idx = torch.zeros(hiddens_unpacked.size(0), 1, hiddens_unpacked.size(2)).long()

        # iterate over each batch
        for i in range(hiddens_unpacked.size(0)):

            # set the second index of the matrix to be the length of each batch
            last_hidden_idx[i, 0, :] = sorted_lengths[i] - 1

        # use cuda if available
        if torch.cuda.is_available():
            last_hidden_idx = last_hidden_idx.cuda()
        
        last_hidden = torch.gather(hiddens_unpacked, 1, Variable(last_hidden_idx, requires_grad=False))
        last_hidden = torch.index_select(last_hidden, 0, Variable(reverse_batch_idx, requires_grad=False))

        return last_hidden


class DecoderLayoutRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderLayoutRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, vocab_size)

        self.init_weights()
    
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
    
    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack(embeddings, lengths, batch_first=True)

        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        sample_ids = []
        inputs = features.unsqueeze(1)

        for _ in range(20):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sample_ids.append(predicted)
            inputs = self.embedding(predicted)
        
        sample_ids = torch.cat(sample_ids, 1)

        return sample_ids.unsqueeze()

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
