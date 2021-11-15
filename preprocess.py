import logging
import random
import torch
from collections import Counter
import gzip
from korean_phonetic_vocab import get_korean_phone_mappings, translate_phone_to_ids

EOS = '<eos>'
BOS = '<bos>'
PAD = '<pad>'
OOV = '<oov>'
BOW = '<bow>'
EOW = '<eow>'
LRB = '-LRB-'
RRB = '-RRB-'

def divide(data, valid_size, gold_tree_list, include_valid_in_train=False, all_train_as_valid=False):
    logging.info('include valid in train is {}; all train as valid is {}'.format(include_valid_in_train, all_train_as_valid))
    assert len(data) == len(gold_tree_list)
    valid_size = min(valid_size, len(data) // 10)
    train_size = len(data) - valid_size
    # random.shuffle(data)
    if include_valid_in_train:
        return data, data[train_size:], gold_tree_list, gold_tree_list[train_size:]
    elif all_train_as_valid:
        return data, data, gold_tree_list, gold_tree_list
    return data[:train_size], data[train_size:], gold_tree_list[:train_size], gold_tree_list[train_size:]

def get_truncated_vocab(dataset, min_count, max_num):

    word_count = Counter()
    for sentence in dataset:
        word_count.update(sentence)

    print(word_count.most_common(10))

    word_count = list(word_count.items())
    word_count.sort(key=lambda x: x[1], reverse=True)

    i = 0
    for word, count in word_count:
        if count < min_count:
            break
        i += 1
    if i > max_num:
        i = max_num

    logging.info('Truncated word count: {}.'.format(sum([count for word, count in word_count[i:]])))
    logging.info('Original vocabulary size: {}. Truncated vocab size {}.'.format(len(word_count), i))
    return word_count[:i]

def break_sentence_with_eos(sentence):
    """
     break sentences with EOS signs to replace break_sentence where it is broken like
     LM
    """
    ret = []
    cur = 0
    start_index = 0
    for index, token in enumerate(sentence):
        if token == EOS:
            end_index = index + 1
            ret.append(sentence[start_index:end_index])
            start_index = end_index
    return ret

def read_corpus(path, korean_phonetics=False):
    """
    read raw text file
    max word len is 28
    """
    data = []
    if path.endswith('gz'):
        fin = gzip.open(path, 'rt', encoding='utf-8')
    else: fin = open(path, 'r', encoding='utf-8')

    if korean_phonetics:
        korean_mapping = get_korean_phone_mappings()
        for line in fin:
            data.append(BOS)
            for token in line.strip().split():
                if len(token) > 28:
                    token = token[:14] + token[-14:]
                token = translate_phone_to_ids(token, korean_mapping)
                data.append(token.lower())
            data.append(EOS)
    else:
        for line in fin:
            data.append(BOS)
            for token in line.strip().split():
                if len(token) > 28:
                    token = token[:14] + token[-14:]
                data.append(token.lower())
            data.append(EOS)
    fin.close()
    logging.info('Longest word in the data:'+str(max([len(s) for s in data])))
    dataset = break_sentence_with_eos(data)
    return dataset


def create_one_batch(x, word2id, char2id, oov=OOV, pad=PAD, sort=True, device='cpu'):

    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    max_len = max(lens)

    if word2id is not None:
        oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
        assert oov_id is not None and pad_id is not None
        batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id).to(device)
        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_w[i][j] = word2id.get(x_ij, oov_id)
    else:
        batch_w = None

    if char2id is not None:
        bow_id, eow_id, oov_id, pad_id = char2id.get(BOW, None), char2id.get(EOW, None), char2id.get(oov, None), char2id.get(pad, None)

        assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None


        max_chars = max([len(w) for i in lst for w in x[i]]) + 2  # counting the <bow> and <eow>

        batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id).to(device)
        batch_var_c = []
        for i, x_i in enumerate(x):
            batch_var_c.append([])
            for j, x_ij in enumerate(x_i):
                if x_ij != BOS and x_ij != EOS:
                    batch_var_c[i].append([])
                    batch_var_c[i][-1].append(bow_id)
                batch_c[i][j][0] = bow_id
                if x_ij == BOS or x_ij == EOS:
                    batch_c[i][j][1] = char2id.get(x_ij)
                    batch_c[i][j][2] = eow_id
                else:

                    for k, c in enumerate(x_ij):
                        batch_c[i][j][k + 1] = char2id.get(c, oov_id)
                        batch_var_c[i][-1].append(char2id.get(c, oov_id))
                    batch_c[i][j][len(x_ij) + 1] = eow_id
                    batch_var_c[i][-1].append(eow_id)

    else:
        batch_c = None
        batch_var_c = None

    for i in range(len(batch_var_c)):
        for j in range(len(batch_var_c[i])):
            batch_var_c[i][j] = torch.tensor(batch_var_c[i][j]).long().to(device)

    masks = [torch.LongTensor(batch_size, max_len).fill_(0).to(device), [], []]

    for i, x_i in enumerate(x):
        for j in range(len(x_i)):
            masks[0][i][j] = 1
            if j + 1 < len(x_i):
                masks[1].append(i * max_len + j)
            if j > 0:
                masks[2].append(i * max_len + j)

    assert len(masks[1]) <= batch_size * max_len
    assert len(masks[2]) <= batch_size * max_len

    masks[1] = torch.LongTensor(masks[1]).to(device)
    masks[2] = torch.LongTensor(masks[2]).to(device)

    return batch_w, batch_c, batch_var_c, lens, masks


# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, eval=False, perm=None, shuffle=True, sort=True, opt=None):

    assert opt is not None
    if eval:
        device = opt.eval_device
    else:
        device = opt.device
    korean_phonetics = opt.korean_phonetics

    lst = perm or list(range(len(x)))
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    sorted_x = [x[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_var_c, batches_lens, batches_masks, batch_indices = [], [], [], [], [], []
    size = batch_size
    cur_len = 0
    start_id = 0
    end_id = 0
    for sorted_index in range(len(sorted_x)):
        if cur_len == 0:
            cur_len = len(sorted_x[sorted_index])
            if len(sorted_x) > 1:
                continue
        if cur_len != len(sorted_x[sorted_index]) or sorted_index - start_id == batch_size or sorted_index == len(sorted_x)-1:
            if sorted_index != len(sorted_x) - 1:
                end_id = sorted_index
            else:
                end_id = None

            if (end_id is None and len(sorted_x[sorted_index]) == cur_len) or end_id is not None:
                bw, bc, batch_var_c, blens, bmasks = create_one_batch(sorted_x[start_id: end_id], word2id, char2id, sort=sort,
                                                         device=device)
                batch_indices.append(lst[start_id:end_id])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_c.append(bc)
                batches_var_c.append(batch_var_c)
                batches_lens.append(blens)
                batches_masks.append(bmasks)
                start_id = end_id
                cur_len = len(sorted_x[sorted_index])
            else:
                end_id = sorted_index
                bw, bc, batch_var_c, blens, bmasks = create_one_batch(sorted_x[start_id: end_id], word2id, char2id, sort=sort,
                                                         device=device)
                batch_indices.append(lst[start_id:end_id])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_c.append(bc)
                batches_var_c.append(batch_var_c)
                batches_lens.append(blens)
                batches_masks.append(bmasks)

                bw, bc, batch_var_c, blens, bmasks = create_one_batch(sorted_x[-1:], word2id, char2id, sort=sort,
                                                         device=device)
                batch_indices.append(lst[-1:])
                sum_len += sum(blens)
                batches_w.append(bw)
                batches_c.append(bc)
                batches_var_c.append(batch_var_c)
                batches_lens.append(blens)
                batches_masks.append(bmasks)

    nbatch = len(batch_indices)

    logging.info("{} batches, avg len: {:.1f}, max len {}, min len {}.".format(nbatch, sum_len / len(x), len(sorted_x[0]),
                                                                               len(sorted_x[-1])))

    if sort:
        perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_var_c = [batches_var_c[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]
    batch_indices = [batch_indices[i] for i in perm]

    return batches_w, batches_c, batches_var_c, batches_lens, batches_masks, batch_indices

def read_markers(fname):
    markers = [0]
    with open(fname) as fh:
        for l in fh:
            marker = int(l.strip().split(' ')[1])
            markers.append(marker)
    return markers