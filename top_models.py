import torch.nn as nn
import torch.distributions
from torch.utils.tensorboard import SummaryWriter
from treenode import convert_binary_matrix_to_strtree

class CharPCFG(nn.Module):
    def __init__(self, pcfg, writer:SummaryWriter=None):
        super(CharPCFG, self).__init__()
        self.pcfg = pcfg
        self.writer = writer
        self.do_parsing = True # type: bool
        self.accumulated_embeddings = None
        self.turn_off_training_for_bilm = False

    def forward(self, word_inp, chars_var_inp, distance_penalty_weight=0.):

        if 'char' in self.pcfg.model_type:
            logprob_list = self.pcfg.forward(chars_var_inp, words=word_inp)
        elif 'word' in self.pcfg.model_type or self.pcfg.model_type == 'subgrams':
            logprob_list = self.pcfg.forward(word_inp)
        else: raise
        total_num_chars = sum([sum([x.numel() for x in y]) for y in chars_var_inp])
        structure_loss = torch.sum(logprob_list, dim=0)

        total_loss = structure_loss
        return total_loss

    def parse(self, word_inp, chars_var_inp, indices, eval=False, set_pcfg=True):

        if self.pcfg.model_type == 'char':
            structure_loss, vtree_list, _, _ = self.pcfg.forward(chars_var_inp, eval, argmax=True,
                                                             indices=indices, set_pcfg=set_pcfg)
        elif self.pcfg.model_type == 'word' or self.pcfg.model_type == 'subgrams':
            structure_loss, vtree_list, _, _ = self.pcfg.forward(word_inp, eval, argmax=True,
                                                             indices=indices, set_pcfg=set_pcfg)
        elif 'compound' in self.pcfg.model_type:
            vtree_list = []
            structure_loss = []
            for index_index, index in enumerate(indices):
                if 'word' in self.pcfg.model_type:
                    sent = word_inp[index_index].unsqueeze(0)
                    local_structure_loss, tree = self.pcfg.forward(sent, argmax=True)
                else:
                    sent = [chars_var_inp[index_index]]
                    words = word_inp[index_index].unsqueeze(0)
                    local_structure_loss, tree = self.pcfg.forward(sent, words=words, argmax=True)
                structure_loss.append(local_structure_loss)
                vtree_list.append(tree)
            structure_loss = torch.cat(structure_loss, dim=0)
        else: raise
        return structure_loss.sum().item(), vtree_list

    def likelihood(self, word_inp, chars_var_inp, indices, set_pcfg=True):

        if self.pcfg.model_type == 'char':
            structure_loss = self.pcfg.forward(chars_var_inp, argmax=False,
                                                             indices=indices, set_pcfg=set_pcfg)
        elif self.pcfg.model_type == 'word' or self.pcfg.model_type == 'subgrams':
            structure_loss = self.pcfg.forward(word_inp, argmax=False,
                                                             indices=indices, set_pcfg=set_pcfg)
        elif 'compound' in self.pcfg.model_type:
            vtree_list = []
            structure_loss = []
            for index_index, index in enumerate(indices):
                if 'word' in self.pcfg.model_type:
                    sent = word_inp[index_index].unsqueeze(0)
                    local_structure_loss, tree = self.pcfg.forward(sent, argmax=False)
                else:
                    sent = [chars_var_inp[index_index]]
                    words = word_inp[index_index].unsqueeze(0)
                    local_structure_loss, tree = self.pcfg.forward(sent, words=words, argmax=False)
                structure_loss.append(local_structure_loss)
                vtree_list.append(tree)
            structure_loss = torch.cat(structure_loss, dim=0)
        else: raise
        return structure_loss.sum().item()

    def skip_parsing(self):
        self.do_parsing = False

    def resume_parsing(self):
        self.do_parsing = True

    def accumulate_embeddings(self, embs):
        self.accumulated_embeddings = embs.detach()

    def repopulate_embeddings(self):

        mean = torch.mean(torch.mean(self.accumulated_embeddings, dim=0), dim=0)
        self.pcfg.nont_emb.data = mean + torch.ones_like(self.pcfg.nont_emb.data).normal_(std=INIT_STD * 2)

        logging.info('Recalibrating the category embeddings...')

    def similarity_penalty(self, embs):
        normed_embs = torch.nn.functional.normalize(embs, dim=2)
        distances = normed_embs @ normed_embs.transpose(-1, -2)
        no_diag = torch.eye(distances.shape[1]).to(distances.device)
        no_diag = (no_diag == 0).float()
        return (distances * no_diag).sum()
