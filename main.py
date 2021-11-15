import os
import sys, gzip
import argparse
import time
import random
import logging
import json
import bidict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from collections import Counter
import numpy as np
from top_models import *
import preprocess
import postprocess, model_use
import model_args
from eval.eval_access import eval_access
from pcfg_models import SimpleCompPCFGCharNoDistinction

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars

def train():

    opt = model_args.parse_args(sys.argv)

    # set seed before anything else.
    if opt.seed < 0: # random seed if seed is set to negative values
        opt.seed = int(int(time.time()) * random.random())
    random_seed(opt.seed, use_cuda=opt.device=='cuda')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfile_fh = gzip.open(os.path.join(opt.model_path, opt.logfile), 'wt')
    writer = SummaryWriter(os.path.join(opt.model_path, 'tensorboard'), flush_secs=10)
    filehandler = logging.StreamHandler(logfile_fh)
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    # Dump configurations
    logging.info(opt)
    writer.add_text('args', str(opt))

    assert (opt.device == 'cuda' and torch.cuda.is_available()) or opt.device == 'cpu'

    train_data = preprocess.read_corpus(opt.train_path, opt.korean_phonetics)

    logging.info('training instance: {}, training tokens: {}.'.format(len(train_data),
                                                                      sum([len(s) - 1 for s in train_data])))

    with open(opt.train_gold_path) as tfh:
        train_tree_list = [x.strip() for x in tfh]

    train_data, valid_data, train_tree_list, valid_tree_list = preprocess.divide(train_data, opt.valid_size, train_tree_list, include_valid_in_train=False,
                                                                                 all_train_as_valid=True) # INCLUDE VALID IN TRAIN TO REDUCE TIME

    logging.info('training instance: {}, training tokens after division: {}.'.format(len(train_data), sum([len(s) - 1 for s in train_data])))
    logging.info('valid instance: {}, valid tokens: {}.'.format(len(valid_data), sum([len(s) - 1 for s in valid_data])))

    if opt.augment_path is not None:
        aug_data = preprocess.read_corpus(opt.augment_path, opt.korean_phonetics)
        training_data_target = opt.augment_target
        augmenting_data_number = training_data_target - len(train_data)
        if augmenting_data_number > 0:
            augmenting_data = aug_data[:augmenting_data_number]
            train_data = train_data + augmenting_data
            logging.info('augmenting data instance: {}, total data instance {}.'.format(len(augmenting_data), training_data_target))
        else:
            train_data = train_data[:training_data_target]
            valid_data = valid_data[:training_data_target]
            valid_tree_list = valid_tree_list[:training_data_target]
            logging.info('reducing data instances to: {} from total data instance {}.'.format(training_data_target, len(train_data)))

    word_lexicon = bidict.bidict()

    # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
    logging.warning('enforcing minimum count of 1')
    opt.min_count = 1
    vocab = preprocess.get_truncated_vocab(train_data, opt.min_count, opt.max_vocab_size)

    # Ensure index of '<oov>' is 0
    special_words = [preprocess.OOV, preprocess.BOS, preprocess.EOS, preprocess.PAD, preprocess.LRB, preprocess.RRB]
    special_chars = [preprocess.BOS, preprocess.EOS, preprocess.OOV, preprocess.PAD, preprocess.BOW, preprocess.EOW]

    for special_word in special_words:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)

    for word, _ in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)

    logging.info('Vocabulary size: {0}'.format(len(word_lexicon)) + '; Max length: {}'.format(max([len(x) for x in word_lexicon])))

    # Character Lexicon
    char_lexicon = bidict.bidict()
    char_grams_lexicon = bidict.bidict()
    for word in special_words:
        char_grams_lexicon[word] = len(char_grams_lexicon)
    if opt.subgram_word:
        for word in word_lexicon:
            if word not in char_grams_lexicon:
                char_grams_lexicon['word ' + word] = len(char_grams_lexicon)
    # add word length feature
    for i in range(1, 1+max([len(x) for x in word_lexicon])):
        char_grams_lexicon['word length '+str(i)] = len(char_grams_lexicon)

    word_indexed_char_grams = {}
    for sentence in train_data:
        for word in sentence:
            if word in word_indexed_char_grams or word in special_words or word not in word_lexicon:
                continue
            else:
                word_indexed_char_grams[word] = set()
                word_indexed_char_grams[word].add(char_grams_lexicon['word length '+str(len(word))])
                if opt.subgram_word:
                    word_indexed_char_grams[word].add(char_grams_lexicon['word ' + word])

            for ch in word:
                if ch not in char_lexicon:
                    char_lexicon[ch] = len(char_lexicon)
            characters = [preprocess.BOW, preprocess.BOW] + list(word) + [preprocess.EOW, preprocess.EOW]
            if opt.subgram_stem and len(characters) - 4 >= 7:
                allfixes = [' '.join(characters[1:-3]), ' '.join(characters[1:-4]), ' '.join(characters[1:-5]),
                            ' '.join(characters[3:-1]), ' '.join(characters[4:-1]), ' '.join(characters[5:-1])]
                for fix in allfixes:
                    if fix not in char_grams_lexicon:
                        char_grams_lexicon[fix] = len(char_grams_lexicon)
                    word_indexed_char_grams[word].add(char_grams_lexicon[fix])
            for index in range(len(characters)-2):
                curgram = ' '.join(characters[index:index + 3])
                if curgram not in char_grams_lexicon:
                    char_grams_lexicon[curgram] = len(char_grams_lexicon)
                word_indexed_char_grams[word].add(char_grams_lexicon[curgram])
            for index in range(1, len(characters)-2):
                curgram = ' '.join(characters[index:index + 2])
                if curgram not in char_grams_lexicon:
                    char_grams_lexicon[curgram] = len(char_grams_lexicon)
                word_indexed_char_grams[word].add(char_grams_lexicon[curgram])

    largest_char_features = max([len(y) for y in word_indexed_char_grams.values()])

    features = []
    offsets = []
    for word_index in range(len(word_lexicon)):
        word = word_lexicon.inv[word_index]
        offsets.append(len(features))
        if word not in word_indexed_char_grams:
            features.append(char_grams_lexicon[word])
        else:
            for val_index, val in enumerate(word_indexed_char_grams[word]):
                features.append(val)

    all_words_char_features = (torch.LongTensor(features), torch.LongTensor(offsets))

    torch.save(all_words_char_features, os.path.join(opt.model_path, 'words_char_features.pth'))

    for special_char in special_chars:
        if special_char not in char_lexicon:
            char_lexicon[special_char] = len(char_lexicon)

    logging.info('Char embedding size: {0}'.format(len(char_lexicon)))
    logging.info('Char Grams size: {0}'.format(len(char_grams_lexicon)))


    train_list = []

    # training batch size for the pre training is 8 times larger than in eval
    train = preprocess.create_batches(train_data, opt.batch_size, word_lexicon, char_lexicon, opt=opt)

    logging.info('Evaluate every {0} epochs.'.format(opt.eval_steps))

    if valid_data is not None:
        valid = preprocess.create_batches(valid_data, opt.batch_size, word_lexicon, char_lexicon, eval=True, opt=opt)
    else:
        valid = None

    logging.info('vocab size: {0}'.format(len(word_lexicon)))

    if opt.model_type not in {"word", "char"}:
        raise ValueError('not recognized model type! {} '.format(opt.model_type))
    else:
        pcfg_parser = SimpleCompPCFGCharNoDistinction(nt_states=opt.num_nonterminal, pret_states=opt.num_preterminal, num_chars=len(char_lexicon),
                                           device=opt.device, eval_device=opt.eval_device, num_words=len(word_lexicon), model_type=opt.model_type,
                                           state_dim=opt.state_dim, char_grams_lexicon=char_grams_lexicon,
                                            all_words_char_features=all_words_char_features, rnn_hidden_dim=opt.rnn_hidden_dim)

    model = CharPCFG(pcfg_parser, writer=writer)

    logging.info(str(model))
    num_grammar_params = 0
    for param in model.parameters():
        num_grammar_params += param.numel()
    logging.info("Top PCFG parser has {} parameters".format(num_grammar_params))

    model = model.to(opt.device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    with open(os.path.join(opt.model_path, 'char.dic'), 'w', encoding='utf-8') as fpo:
        for ch, i in char_lexicon.items():
            print('{0}\t{1}'.format(ch, i), file=fpo)

    with open(os.path.join(opt.model_path, 'word.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in word_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    with open(os.path.join(opt.model_path, 'subgrams.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in char_grams_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    opt_save_path = os.path.join(opt.model_path, 'opt.pth')
    torch.save(opt, opt_save_path)

    best_eval_likelihood = -1e+8

    patient = 0

    if opt.checkpoint != "":
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint)

    for epoch in range(opt.start_epoch, opt.max_epoch):

        if train_list and epoch >= len(train_list):
            break  # stop if doing the childes marker thing
        elif train_list and epoch < len(train_list):
            train = train_list[epoch]
        else:
            pass

        optimizer = model_use.train_model(epoch, opt, model, optimizer, train)

        if ((epoch - opt.eval_start_epoch) % opt.eval_steps == 0 or epoch + 1 == opt.max_epoch) and epoch >= opt.eval_start_epoch:

            logging.info('EVALING.')

            if opt.eval_parsing:
                # evaluate on CPU
                model.to(opt.eval_device)

                total_eval_likelihoods, trees = model_use.parse_dataset(model, valid, epoch)
                total_eval_likelihoods = total_eval_likelihoods

                tree_fn, valid_pred_trees = postprocess.print_trees(trees, valid_data, epoch, opt)
                eval_access(valid_pred_trees, valid_tree_list, model.writer, epoch)

                # back to GPU for training
                model.to(opt.device)

            else:
                total_eval_likelihoods = model_use.likelihood_dataset(model, valid, epoch) * (-1)

            if total_eval_likelihoods > best_eval_likelihood:
                logging.info('Better model found based on likelihood: {}! vs {}'.format(total_eval_likelihoods, best_eval_likelihood))
                best_eval_likelihood = total_eval_likelihoods
                patient = 0
                model_save_path = os.path.join(opt.model_path, 'model.pth')
                torch.save(model.state_dict(), model_save_path)

            else:
                patient += 1
                if patient >= opt.eval_patient:
                    break

    model.writer.close()

def test():

    opt = model_args.parse_args(sys.argv)

    # set seed before anything else.
    if opt.seed < 0: # random seed if seed is set to negative values
        opt.seed = int(int(time.time()) * random.random())
    random_seed(opt.seed, use_cuda=opt.device=='cuda')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logfile_fh = gzip.open(os.path.join(opt.model_path, opt.logfile), 'wt')
    writer = SummaryWriter(os.path.join(opt.model_path, 'tensorboard'), flush_secs=10)
    filehandler = logging.StreamHandler(logfile_fh)
    streamhandler = logging.StreamHandler(sys.stdout)
    handler_list = [filehandler, streamhandler]
    logging.basicConfig(level='INFO', format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', handlers=handler_list)

    # Dump configurations
    logging.info(opt)
    writer.add_text('args', str(opt))

    assert (opt.device == 'cuda' and torch.cuda.is_available()) or opt.device == 'cpu'

    train_data = preprocess.read_corpus(opt.train_path, opt.korean_phonetics)

    logging.info('training instance: {}, training tokens: {}.'.format(len(train_data),
                                                                      sum([len(s) - 1 for s in train_data])))

    with open(opt.train_gold_path) as tfh:
        train_tree_list = [x.strip() for x in tfh]

    train_data, valid_data, train_tree_list, valid_tree_list = preprocess.divide(train_data, opt.valid_size, train_tree_list, include_valid_in_train=False,
                                                                                 all_train_as_valid=True) # INCLUDE VALID IN TRAIN TO REDUCE TIME

    logging.info('training instance: {}, training tokens after division: {}.'.format(len(train_data), sum([len(s) - 1 for s in train_data])))
    logging.info('valid instance: {}, valid tokens: {}.'.format(len(valid_data), sum([len(s) - 1 for s in valid_data])))

    word_lexicon = bidict.bidict()

    # Maintain the vocabulary. vocabulary is used in either WordEmbeddingInput or softmax classification
    logging.warning('enforcing minimum count of 1')
    opt.min_count = 1
    vocab = preprocess.get_truncated_vocab(train_data, opt.min_count, opt.max_vocab_size)

    # Ensure index of '<oov>' is 0
    special_words = [preprocess.OOV, preprocess.BOS, preprocess.EOS, preprocess.PAD, preprocess.LRB, preprocess.RRB]
    special_chars = [preprocess.BOS, preprocess.EOS, preprocess.OOV, preprocess.PAD, preprocess.BOW, preprocess.EOW]

    for special_word in special_words:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)

    for word, _ in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)

    logging.info('Vocabulary size: {0}'.format(len(word_lexicon)) + '; Max length: {}'.format(max([len(x) for x in word_lexicon])))

    # Character Lexicon
    char_lexicon = bidict.bidict()
    char_grams_lexicon = bidict.bidict()
    for word in special_words:
        char_grams_lexicon[word] = len(char_grams_lexicon)
    if opt.subgram_word:
        for word in word_lexicon:
            if word not in char_grams_lexicon:
                char_grams_lexicon['word ' + word] = len(char_grams_lexicon)

    # add word length feature
    for i in range(1, 1+max([len(x) for x in word_lexicon])):
        char_grams_lexicon['word length '+str(i)] = len(char_grams_lexicon)

    word_indexed_char_grams = {}
    for sentence in train_data:
        for word in sentence:
            if word in word_indexed_char_grams or word in special_words:
                continue
            else:
                word_indexed_char_grams[word] = set()
                word_indexed_char_grams[word].add(char_grams_lexicon['word length '+str(len(word))])
                if opt.subgram_word:
                    word_indexed_char_grams[word].add(char_grams_lexicon['word ' + word])

            for ch in word:
                if ch not in char_lexicon:
                    char_lexicon[ch] = len(char_lexicon)
            characters = [preprocess.BOW, preprocess.BOW] + list(word) + [preprocess.EOW, preprocess.EOW]
            if opt.subgram_stem and len(characters) - 4 >= 7:
                allfixes = [' '.join(characters[1:-3]), ' '.join(characters[1:-4]), ' '.join(characters[1:-5]),
                            ' '.join(characters[3:-1]), ' '.join(characters[4:-1]), ' '.join(characters[5:-1])]
                for fix in allfixes:
                    if fix not in char_grams_lexicon:
                        char_grams_lexicon[fix] = len(char_grams_lexicon)
                    word_indexed_char_grams[word].add(char_grams_lexicon[fix])
            for index in range(len(characters)-2):
                curgram = ' '.join(characters[index:index + 3])
                if curgram not in char_grams_lexicon:
                    char_grams_lexicon[curgram] = len(char_grams_lexicon)
                word_indexed_char_grams[word].add(char_grams_lexicon[curgram])
            for index in range(1, len(characters)-2):
                curgram = ' '.join(characters[index:index + 2])
                if curgram not in char_grams_lexicon:
                    char_grams_lexicon[curgram] = len(char_grams_lexicon)
                word_indexed_char_grams[word].add(char_grams_lexicon[curgram])

    largest_char_features = max([len(y) for y in word_indexed_char_grams.values()])

    features = []
    offsets = []
    for word_index in range(len(word_lexicon)):
        word = word_lexicon.inv[word_index]
        offsets.append(len(features))
        if word not in word_indexed_char_grams:
            features.append(char_grams_lexicon[word])
        else:
            for val_index, val in enumerate(word_indexed_char_grams[word]):
                features.append(val)

    all_words_char_features = (torch.LongTensor(features), torch.LongTensor(offsets))

    torch.save(all_words_char_features, os.path.join(opt.model_path, 'words_char_features.pth'))

    for special_char in special_chars:
        if special_char not in char_lexicon:
            char_lexicon[special_char] = len(char_lexicon)

    logging.info('Char embedding size: {0}'.format(len(char_lexicon)))
    logging.info('Char Grams size: {0}'.format(len(char_grams_lexicon)))
    logging.info('Evaluate every {0} epochs.'.format(opt.eval_steps))

    if valid_data is not None:
        valid = preprocess.create_batches(valid_data, opt.batch_size, word_lexicon, char_lexicon, eval=True, opt=opt)
    else:
        valid = None

    logging.info('vocab size: {0}'.format(len(word_lexicon)))

    if opt.model_type not in {"word", "char"}:
        raise ValueError('not recognized model type! {} '.format(opt.model_type))
    else:
        pcfg_parser = SimpleCompPCFGCharNoDistinction(nt_states=opt.num_nonterminal, pret_states=opt.num_preterminal, num_chars=len(char_lexicon),
                                           device=opt.device, eval_device=opt.eval_device, num_words=len(word_lexicon), model_type=opt.model_type,
                                           state_dim=opt.state_dim, char_grams_lexicon=char_grams_lexicon,
                                            all_words_char_features=all_words_char_features, rnn_hidden_dim=opt.rnn_hidden_dim)

    model = CharPCFG(pcfg_parser, writer=writer)

    logging.info(str(model))
    num_grammar_params = 0
    for param in model.parameters():
        num_grammar_params += param.numel()
    logging.info("Top PCFG parser has {} parameters".format(num_grammar_params))

    model = model.to(opt.device)

    with open(os.path.join(opt.model_path, 'char.dic'), 'w', encoding='utf-8') as fpo:
        for ch, i in char_lexicon.items():
            print('{0}\t{1}'.format(ch, i), file=fpo)

    with open(os.path.join(opt.model_path, 'word.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in word_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    with open(os.path.join(opt.model_path, 'subgrams.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in char_grams_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    opt_save_path = os.path.join(opt.model_path, 'opt.pth')
    torch.save(opt, opt_save_path)

    if opt.checkpoint != "":
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint)
        logging.info('Model loaded from {}.'.format(opt.checkpoint))

    logging.info('EVALING.')

    if opt.eval_parsing:
        # evaluate on CPU
        if opt.device != opt.eval_device:
            model.to(opt.eval_device)

        _, trees = model_use.parse_dataset(model, valid, 0)

        tree_fn, valid_pred_trees = postprocess.print_trees(trees, valid_data, 0, opt)
        eval_access(valid_pred_trees, valid_tree_list, model.writer, 0)

        # back to GPU for training
        if opt.device != opt.eval_device:
            model.to(opt.device)

    model.writer.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
        logging.shutdown()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
