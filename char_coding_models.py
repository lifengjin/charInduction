import torch
import torch.nn as nn
from torch.nn import functional as F


class ResidualLayer(nn.Module): # from kim
    def __init__(self, in_dim=100,
                 out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class CharProbRNN(nn.Module):

    def __init__(self, num_chars, state_dim=256, hidden_size=256, num_layers=4, dropout=0.):
        super(CharProbRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.state_dim = state_dim

        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        # self.rnn = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

        self.top_fc = nn.Linear(hidden_size, num_chars)
        self.char_embs = nn.Embedding(num_chars, hidden_size)

        # self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size), nn.ReLU())
        self.cat_emb_expansion = nn.Sequential(nn.Linear(state_dim, hidden_size*num_layers), nn.ReLU())

        torch.nn.init.kaiming_normal_(self.char_embs.weight.data)

    def forward(self, chars, cat_embs, set_pcfg=True): # does not use set pcfg
        char_embs, cat_embs = self.prep_input(chars, cat_embs)
        Hs = []
        lens = 0
        for cat_tensor in cat_embs: # each cat at one time
            # for simple RNNs
            # # cat_tensor is batch, dim
            # cat_tensor = cat_tensor.unsqueeze(0).expand(self.num_layers, -1, -1)
            # cat_tensor = self.cat_emb_expansion(cat_tensor)
            # all_hs, _ = self.rnn.forward(char_embs, cat_tensor)
            # all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            # Hs.append(all_hs[0].transpose(0,1))
            # lens = all_hs[1]

            # for LSTMs with 3d linears
            cat_tensor = self.cat_emb_expansion(cat_tensor) # batch, hidden*numlayers
            cat_tensor = cat_tensor.reshape(cat_tensor.shape[0], self.hidden_size, -1)
            cat_tensor = cat_tensor.permute(2, 0, 1)
            h0_tensor = torch.zeros_like(cat_tensor)
            all_hs, _ = self.rnn.forward(char_embs, (h0_tensor, cat_tensor))
            all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            Hs.append(all_hs[0].transpose(0,1))
            lens = all_hs[1]

        Hses = torch.stack(Hs, 0)
        # Hses = nn.functional.relu(Hses)
        scores = self.top_fc.forward(Hses) # cats, batch, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        total_logprobs = []

        for idx, length in enumerate(lens.tolist()):
            this_word_logprobs = logprobs[:, idx, :length, :] # cats, (batch_scalar), num_chars_in_word, num_chars
            sent_id = idx // len(chars[0])
            word_id = idx % len(chars[0])
            targets = chars[sent_id][word_id][1:]
            this_word_logprobs = this_word_logprobs[:, range(this_word_logprobs.shape[1]), targets]  # cats, num_chars_in_word
            total_logprobs.append(this_word_logprobs.sum(-1)) # cats
        total_logprobs = torch.stack(total_logprobs, dim=0) # batch, cats
        total_logprobs = total_logprobs.reshape(len(chars), -1, total_logprobs.shape[1]) # sentbatch, wordbatch, cats
        # total_logprobs = total_logprobs.transpose(0, 1) # wordbatch, sentbatch, cats
        return total_logprobs

    def prep_input(self, chars, cat_embs):
        # cat_embs is num_cat, cat_dim
        # chars is num_words, word/char_tensor
        embeddings = []
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                embeddings.append(self.char_embs.forward(chars[i][j][:-1])) # no word end token
        packed_char_embs = nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False) # len, batch, embs
        expanded_cat_embs = cat_embs.unsqueeze(1).expand(-1, packed_char_embs.data.size(0), -1) # numcat,batch, catdim

        return packed_char_embs, expanded_cat_embs

class CharProbRNNCategorySpecific(nn.Module):

    def __init__(self, num_chars, state_dim, num_layers=3, dropout=0., num_t=10):
        super(CharProbRNNCategorySpecific, self).__init__()
        self.num_layers = num_layers
        self.rnns = nn.ModuleDict()
        self.fcs = nn.ModuleDict()
        self.num_t = num_t
        for i in range(num_t):
            self.rnns[str(i)] = nn.RNN(state_dim, state_dim, num_layers=num_layers, dropout=dropout)
            self.fcs[str(i)] = nn.Linear(state_dim, num_chars)
        self.char_embs = nn.Embedding(num_chars, state_dim)
        torch.nn.init.kaiming_normal_(self.char_embs.weight.data)

    def forward(self, chars, cat_embs, set_pcfg=True): # does not use set pcfg
        char_embs = self.prep_input(chars)
        scores = []
        lens = 0
        for i in range(self.num_t): # each cat at one time
            # cat_tensor is batch, dim
            rnn = self.rnns[str(i)]
            fc = self.fcs[str(i)]
            all_hs, _ = rnn.forward(char_embs)
            all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            probs = fc(all_hs[0].transpose(0,1))
            scores.append(probs)
            lens = all_hs[1]
        scores = torch.stack(scores, 0)
        # scores = self.top_fc.forward(Hses) # cats, batch, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        total_logprobs = []

        for idx, length in enumerate(lens.tolist()):
            this_word_logprobs = logprobs[:, idx, :length, :] # cats, (batch_scalar), num_chars_in_word, num_chars
            sent_id = idx // len(chars[0])
            word_id = idx % len(chars[0])
            targets = chars[sent_id][word_id][1:]
            this_word_logprobs = this_word_logprobs[:, range(this_word_logprobs.shape[1]), targets]  # cats, num_chars_in_word
            total_logprobs.append(this_word_logprobs.sum(-1)) # cats
        total_logprobs = torch.stack(total_logprobs, dim=0) # batch, cats
        total_logprobs = total_logprobs.reshape(len(chars), -1, total_logprobs.shape[1]) # sentbatch, wordbatch, cats
        # total_logprobs = total_logprobs.transpose(0, 1) # wordbatch, sentbatch, cats
        return total_logprobs

    def prep_input(self, chars):
        # cat_embs is num_cat, cat_dim
        # chars is num_words, word/char_tensor
        embeddings = []
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                embeddings.append(self.char_embs.forward(chars[i][j][:-1])) # no word end token
        packed_char_embs = nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False) # len, batch, embs

        return packed_char_embs

class CharProbRNNFixVocab(nn.Module):
    #TODO
    def __init__(self, num_chars, state_dim, num_layers=3, dropout=0.5):
        super(CharProbRNNFixVocab, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.RNN(state_dim, state_dim, num_layers=num_layers, dropout=dropout)
        self.top_fc = nn.Linear(state_dim, num_chars)
        self.char_embs = nn.Embedding(num_chars, state_dim)

    def forward(self, chars, cat_embs):
        char_embs, cat_embs = self.prep_input(chars, cat_embs)
        Hs = []
        lens = 0
        for cat_tensor in cat_embs: # each cat at one time
            # cat_tensor is batch, dim
            cat_tensor = cat_tensor.unsqueeze(0).expand(self.num_layers, -1, -1)
            all_hs, _ = self.rnn.forward(char_embs, cat_tensor)
            all_hs = nn.utils.rnn.pad_packed_sequence(all_hs) # len, batch, embs
            Hs.append(all_hs[0].transpose(0,1))
            lens = all_hs[1]
        Hses = torch.stack(Hs, 0)
        scores = self.top_fc.forward(Hses) # cats, batch, num_chars_in_word, num_chars
        logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        total_logprobs = []

        for idx, length in enumerate(lens.tolist()):
            this_word_logprobs = logprobs[:, idx, :length, :] # cats, (batch_scalar), num_chars_in_word, num_chars
            sent_id = idx // len(chars[0])
            word_id = idx % len(chars[0])
            targets = chars[sent_id][word_id][1:]
            this_word_logprobs = this_word_logprobs[:, range(this_word_logprobs.shape[1]), targets]  # cats, num_chars_in_word
            total_logprobs.append(this_word_logprobs.sum(-1)) # cats
        total_logprobs = torch.stack(total_logprobs, dim=0) # batch, cats
        total_logprobs = total_logprobs.reshape(len(chars), -1, total_logprobs.shape[1]) # sentbatch, wordbatch, cats
        return total_logprobs

    def prep_input(self, chars, cat_embs):
        # cat_embs is num_cat, cat_dim
        # chars is num_words, word/char_tensor
        embeddings = []
        for i in range(len(chars)):
            for j in range(len(chars[i])):
                embeddings.append(self.char_embs.forward(chars[i][j][:-1])) # no word end token
        packed_char_embs = nn.utils.rnn.pack_sequence(embeddings, enforce_sorted=False) # len, batch, embs
        expanded_cat_embs = cat_embs.unsqueeze(1).expand(-1, packed_char_embs.data.size(0), -1) # numcat,batch, catdim

        return packed_char_embs, expanded_cat_embs

class WordProbFCFixVocab(nn.Module):
    def __init__(self, num_words, state_dim, dropout=0.0):
        super(WordProbFCFixVocab, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim*2, state_dim*2),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(state_dim*2, state_dim*2),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(state_dim*2, 1))
        self.word_embs_module = nn.Embedding(num_words, state_dim)
        self.word_embs = self.word_embs_module.weight

    def forward(self, words, cat_embs, set_pcfg=True):
        if set_pcfg:
            dist = self.prep_input(cat_embs)
            self.dist = dist
        else:
            pass
        word_indices = words[:, 1:-1]

        logprobs = self.dist[word_indices, :] # sent, word, cats; get rid of bos and eos
        return logprobs

    def prep_input(self, cat_embs):
        # cat_embs is num_cat, cat_dim
        expanded_embs = self.word_embs # words, dim
        cat_embs = cat_embs # cats, dim

        outs = []
        for cat_emb in cat_embs:
            cat_emb = cat_emb.unsqueeze(0).expand(self.word_embs.shape[0], -1) # words, dim

            inp = torch.cat([expanded_embs, cat_emb], dim=-1) # words, dim*2
            out = self.fc(inp) # vocab, 1
            outs.append(out)
        outs = torch.cat(outs, dim=-1) # vocab, cats
        logprobs = nn.functional.log_softmax(outs, dim=0) # vocab, cats

        return logprobs

class WordProbFCFixVocabCompound(nn.Module):
    def __init__(self, num_words, state_dim, dropout=0.0):
        super(WordProbFCFixVocabCompound, self).__init__()
        self.fc = nn.Sequential(nn.Linear(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, num_words))

    def forward(self, words, cat_embs, set_pcfg=True):
        if set_pcfg:
            dist = nn.functional.log_softmax(self.fc(cat_embs), 1).t() # vocab, cats
            self.dist = dist
        else:
            pass
        word_indices = words[:, 1:-1]

        logprobs = self.dist[word_indices, :] # sent, word, cats; get rid of bos and eos
        return logprobs

class CharProbLogistic(nn.Module):
    def __init__(self, char_grams_lexicon, all_words_char_features, num_t=75):
        super(CharProbLogistic, self).__init__()
        self.register_buffer('all_words_char_features', all_words_char_features[0])
        self.register_buffer('offsets', all_words_char_features[1])
        self.num_char_features = len(char_grams_lexicon)
        self.char_grams_lexicon = char_grams_lexicon
        self.num_t = num_t
        self.num_words = len(all_words_char_features[1])
        self.logistic = nn.EmbeddingBag(self.num_char_features, self.num_t, mode='sum')
        # self.all_words_char_features = all_words_char_features.unsqueeze(0).expand(self.num_t, -1, -1).reshape(-1, all_words_char_features.shape[-1]) # num_t * vocab, features

    def forward(self, words):
        words = words[:, 1:-1]

        logits = self.logistic(self.all_words_char_features, self.offsets)# vocab, num_t
        logprobs = F.log_softmax(logits, dim=0) # vocab, num_t

        total_logprobs = logprobs[words, :] # sent, word, num_t

        return total_logprobs