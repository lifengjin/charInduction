import argparse
import logging
import bidict, os, sys
import preprocess
import torch
from top_models import *
from pcfg_models import SimpleCompPCFG, SimpleCompPCFGChar
import model_use
import postprocess
from eval.eval_access import eval_access

parser = argparse.ArgumentParser()

parser.add_argument('--model-path', required=True)
parser.add_argument('--toks-path', required=True)
parser.add_argument('--gold-path', required=True)
parser.add_argument('--device', default='cuda')
parser.add_argument('--prefix', default='-eval', help='use this for eval in the same folder with different datasets')

args = parser.parse_args()


logfile_fh = open(os.path.join(args.model_path, 'eval.results'), 'w')
streamhandler = logging.StreamHandler(sys.stdout)
filehandler = logging.StreamHandler(logfile_fh)
handler_list = [filehandler, streamhandler]
logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

model_opt = torch.load(os.path.join(args.model_path, 'opt.pth'))
logging.info('Model folder is: {}'.format(args.model_path))
logging.info('Eval set is: {}'.format(args.toks_path))
logging.info('Gold set is: {}'.format(args.gold_path))

char_lexicon = bidict.bidict()
word_lexicon = bidict.bidict()
char_grams_lexicon = bidict.bidict()

with open(os.path.join(args.model_path, 'char.dic'), encoding='utf-8') as fpo:
    for line in fpo:
        ch, i = line.strip().split('\t')
        char_lexicon[ch] = int(i)

with open(os.path.join(args.model_path, 'word.dic'), encoding='utf-8') as fpo:

    for line in fpo:
        w, i = line.strip().split('\t')
        word_lexicon[w] = int(i)

with open(os.path.join(args.model_path, 'subgrams.dic'), encoding='utf-8') as fpo:
    for line in fpo:
        w, i = line.strip().split('\t')
        char_grams_lexicon[w] = int(i)

with open(args.gold_path) as tfh:
    parse_tree_list = [x.strip() for x in tfh]

parse_toks = preprocess.read_corpus(args.toks_path)
parse_patches = preprocess.create_batches(
            parse_toks, 1, word_lexicon, char_lexicon,  device=args.device)

logging.info('Word vocab size: {}'.format(len(word_lexicon)))
logging.info('Char vocab size: {}'.format(len(char_lexicon)))
logging.info('Chargram vocab size: {}'.format(len(char_grams_lexicon)))


if 'compound' not in model_opt.model_type:

    all_words_char_features = torch.load(os.path.join(args.model_path, 'words_char_features.pth'))

    pcfg_parser = SimpleCompPCFGCharNoDistinction(nt_states=model_opt.num_nonterminal, pret_states=model_opt.num_preterminal,
                                                  num_chars=len(char_lexicon),
                                                  device=args.device, num_words=len(word_lexicon), model_type=model_opt.model_type,
                                                  state_dim=model_opt.state_dim, char_grams_lexicon=char_grams_lexicon,
                                                  all_words_char_features=all_words_char_features)

elif model_opt.model_type == 'compound-word':
    pcfg_parser = SimpleCompPCFG(vocab=len(word_lexicon), state_dim=model_opt.state_dim, t_states=model_opt.num_preterminal,
                                 nt_states=model_opt.num_nonterminal)
elif model_opt.model_type == 'compound-char':
    char_rnn_type = model_opt.char_rnn_type
    pcfg_parser = SimpleCompPCFGChar(num_chars=len(char_lexicon), state_dim=model_opt.state_dim, t_states=model_opt.num_preterminal,
                                     nt_states=model_opt.num_nonterminal, char_rnn_type=char_rnn_type)
else:
    raise ValueError('not recognized model type! {} '.format(model_opt.model_type))

logging.info('Model type is: {}'.format(model_opt.model_type))
logging.info('Eval corpus size: {}'.format(len(parse_toks)))

model = CharPCFG(pcfg_parser, model_opt.structure_loss_weight)
if args.device == 'cpu':
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'model.pth'), map_location=args.device))
else:
    model.load_state_dict(torch.load(os.path.join(args.model_path, 'model.pth')))
    model.to(args.device)

total_eval_likelihoods, trees = model_use.parse_dataset(model, parse_patches, 'eval')
#
# for index, t in enumerate(trees):
#     if len(t.leaves()) != len(parse_toks[index]):
#         print(index, len(t.leaves()), len(parse_toks[index]))

logging.info('Total likelihood for valid: {}'.format(total_eval_likelihoods))

tree_fn, valid_pred_trees = postprocess.print_trees(trees, parse_toks, args.prefix, model_opt)
eval_access(valid_pred_trees, parse_tree_list, model.writer, args.prefix)
