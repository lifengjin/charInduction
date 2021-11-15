import torch
from torch import nn
import torch.nn.functional as F
from cky_parser_sgd import batch_CKY_parser, SMALL_NEGATIVE_NUMBER
from treenode import convert_binary_matrix_to_strtree
from char_coding_models import CharProbRNN, WordProbFCFixVocab, CharProbRNNCategorySpecific, ResidualLayer, WordProbFCFixVocabCompound, CharProbLogistic


class SimpleCompPCFGCharNoDistinction(nn.Module):
    def __init__(self,
                 state_dim=64,
                 pret_states=30,
                 nt_states=60,
                 num_chars=100,
                 device='cpu',
                 eval_device="cpu",
                 model_type='char',
                 num_words=100,
                 char_grams_lexicon=None,
                 all_words_char_features=None,
                 rnn_hidden_dim=320):
        super(SimpleCompPCFGCharNoDistinction, self).__init__()
        self.state_dim = state_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.model_type = model_type
        self.all_states = nt_states + pret_states
        if self.model_type == 'char':
            self.emit_prob_model = CharProbRNN(num_chars, state_dim=self.state_dim, hidden_size=rnn_hidden_dim)
        elif self.model_type == 'word':
            self.emit_prob_model = WordProbFCFixVocabCompound(num_words, state_dim)
        elif self.model_type == 'subgrams':
            self.emit_prob_model = CharProbLogistic(char_grams_lexicon, all_words_char_features, num_t=self.all_states)
        self.nont_emb = nn.Parameter(torch.randn(self.all_states, state_dim))

        self.rule_mlp = nn.Linear(state_dim, self.all_states ** 2)

        self.root_emb = nn.Parameter(torch.randn(1, state_dim))
        self.root_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      ResidualLayer(state_dim, state_dim),
                                      nn.Linear(state_dim, self.all_states))

        self.split_mlp = nn.Sequential(nn.Linear(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       ResidualLayer(state_dim, state_dim),
                                       nn.Linear(state_dim, 2))

        self.device = device
        self.eval_device = eval_device
        self.pcfg_parser = batch_CKY_parser(nt=self.all_states, t=0, device=self.device)

    def forward(self, x, eval=False, argmax=False, use_mean=False, indices=None, set_pcfg=True, return_ll=True, **kwargs):
        # x : batch x n
        if set_pcfg:
            self.emission = None

            nt_emb = self.nont_emb

            root_scores = F.log_softmax(self.root_mlp(self.root_emb).squeeze(), 0)
            full_p0 = root_scores

            # rule_score = F.log_softmax(self.rule_mlp([nt_emb, nt_emb, nt_emb]).squeeze().reshape([self.all_states, self.all_states**2]), dim=1)
            rule_score = F.log_softmax(self.rule_mlp(nt_emb), 1)  # nt x t**2

            full_G = rule_score
            split_scores = F.log_softmax(self.split_mlp(nt_emb), dim=1)
            full_G = full_G + split_scores[:, 0][..., None]

            self.pcfg_parser.set_models(full_p0, full_G, self.emission, pcfg_split=split_scores)

        if self.model_type != 'subgrams':
            x = self.emit_prob_model(x, self.nont_emb, set_pcfg=set_pcfg)
        else:
            x = self.emit_prob_model(x)

        if argmax:
            if eval and self.device != self.eval_device:
                print("Moving model to {}".format(self.eval_device))
                self.pcfg_parser.device = self.eval_device
            with torch.no_grad():
                logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list = \
                    self.pcfg_parser.marginal(x, viterbi_flag=True, only_viterbi=not return_ll, sent_indices=indices)
            if eval and self.device != self.eval_device:
                self.pcfg_parser.device = self.device
                print("Moving model back to {}".format(self.device))
            return logprob_list, vtree_list, vproduction_counter_dict_list, vlr_branches_list
        else:
            logprob_list, _, _, _ = self.pcfg_parser.marginal(x)
            logprob_list = logprob_list * (-1)
            return logprob_list
