import gzip, multiprocessing
import nltk
from sklearn.metrics import homogeneity_completeness_v_measure
import argparse
from .fix_terminals_wsj import single_fix_terms
from itertools import chain
import numpy as np
from collections import Counter
import logging
from copy import deepcopy
############
# PIOC files must be fix-terminal-ed first.
############

def eval(gold_pred):

    gt = gold_pred[0]
    pt = gold_pred[1]

    if pt.label() == 'x':
        return 0, 0, 0, 0, [], [], 0, Counter(), Counter(), Counter(), Counter()

    g_spans = []
    p_spans = []
    gold_labels = []
    pred_labels = []
    matching_gold_labels = []
    matching_pred_labels = []
    matching_labeled_consts = Counter()
    matching_cross_labeled_consts = Counter()
    gold_labeled_counts = Counter()
    # all_gold_label_counts = Counter()
    all_pred_label_counts = Counter()
    assert len(gt.leaves()) == len(pt.leaves()), "{}\n {}".format(gt, pt)
    # print(gt)
    for subtree in gt.subtrees(lambda x: x.height() > 2):
        g_spans.append(' '.join(subtree.leaves()))
        this_gold_labels = subtree.label().split('+')
        if this_gold_labels[0] == '':
            if len(this_gold_labels) > 1:
                chosen_label = this_gold_labels[1]
            else:
                chosen_label = 'S'  # this is special for negra
        else:
            chosen_label = this_gold_labels[0]
        if '-' in chosen_label:
            chosen_label = chosen_label.split('-')[0]
        gold_labels.append(chosen_label)
        gold_labeled_counts.update([chosen_label])

    for subtree in pt.subtrees(lambda x: x.height() > 2):
        p_spans.append(' '.join(subtree.leaves()))
        pred_labels.append(subtree.label().split('+')[0])

    ggt = gt.copy(deep=True)
    ppt = pt.copy(deep=True)

    this_total_gold_spans = len(g_spans)
    this_total_predicted_spans = len(p_spans)
    this_correct_spans = 0
    len_gold = len(g_spans)
    if len_gold == 0:
        # print('Sent', index, 'Single word sent!', gt)
        return this_total_gold_spans, this_total_predicted_spans, 1, 0, [], [], 1, Counter(), Counter(), Counter(), Counter()
    len_predicted = len(p_spans)
    all_pred_label_counts.update(pred_labels)
    gg_spans = g_spans[:]
    ggold_labels = gold_labels[:]
    # p_spans: concatenation of leaves
    # this_correct_spans could be incorrect (not sensitive to span position)
    # probably won't make a difference, but check if there are no edge cases
    # EVALB can always be used to report scores...
    for span, span_label in zip(p_spans, pred_labels):
        if span in g_spans:
            this_correct_spans += 1
            matching_pred_labels.append(span_label)
            g_span_index = g_spans.index(span)
            matching_gold_labels.append(gold_labels[g_span_index])
            matching_labeled_consts.update([gold_labels[g_span_index]])
            del g_spans[g_span_index]
            del gold_labels[g_span_index]

    # replaces leaves with integers for integer span
    i = 0
    for g_leaf, p_leaf in zip(ggt.treepositions(order='leaves'), ppt.treepositions(order='leaves')):
        ggt[g_leaf] = str(i)
        ppt[p_leaf] = str(i)
        i += 1

    gg_int_spans = []
    for subtree in ggt.subtrees(lambda x: x.height() > 2):
        gg_int_spans.append(' '.join(subtree.leaves()))
    pp_int_spans = []
    for subtree in ppt.subtrees(lambda x: x.height() > 2):
        pp_int_spans.append(' '.join(subtree.leaves()))

    for span in pp_int_spans:
        if span not in gg_int_spans:
            start, end = span[0], span[-1]
            for gspan in gg_int_spans:
                # finding spans that are either a superset or a subset of a gold span
                if (start in gspan and gspan[0] != start and end not in gspan) or (start not in gspan and end in gspan and
                gspan[-1] != end):
                    break
            else:
                cross_span_index = pp_int_spans.index(span)
                matching_cross_labeled_consts.update([pred_labels[cross_span_index]])

    this_r = this_correct_spans / len_gold
    this_p = this_correct_spans / len_predicted
    this_f = 2 * (this_p * this_r / (this_p + this_r + 1e-6))
    this_words = len(gt.leaves())
    return this_total_gold_spans, this_total_predicted_spans, 0, this_correct_spans, matching_gold_labels, \
           matching_pred_labels, this_words, gold_labeled_counts, matching_labeled_consts, \
           matching_cross_labeled_consts, all_pred_label_counts


PUNC_LABELS = {'PUNC', 'PU', 'PONC', 'PUNCT', "sf", ".", ",", "$.", "$,"}

def delete_punc(original_t, do_nothing=0, punc_indices=None):
    # assert (return_punc_indices and not punc_indices) or (not return_punc_indices and punc_indices is not None)
    t = deepcopy(original_t)
    if not do_nothing:
        if punc_indices is not None:
            return_punc_indices = False
        else:
            return_punc_indices = True

        t = nltk.ParentedTree.convert(t)
        indices = []
        indexed_subs = list(enumerate(t.subtrees(filter=lambda x: x.height() == 2)))
        for sub_index, sub in indexed_subs:
            # print(sub_index, sub)

            if sub.label() in PUNC_LABELS or 'PUNC' in sub.label() or (return_punc_indices is False and sub_index in punc_indices):  #
                parent = sub.parent()
                # print(sub_index, sub)
                while parent and len(parent) == 1:
                    sub = parent
                    parent = sub.parent()
                try:
                    del t[sub.treeposition()]
                except:
                    print(t)
                    print(t[sub.treeposition()])
                    raise
                if return_punc_indices:
                    indices.append(sub_index)

        t = nltk.Tree.convert(t)
        t.collapse_unary(collapsePOS=True, collapseRoot=True)
        if return_punc_indices:
            return t, indices
        return t, []
    else:
        t.collapse_unary(collapsePOS=True, collapseRoot=True)
    return t, []

#
# def delete_punc(t):
#     t = nltk.ParentedTree.convert(t)
#     for sub in reversed(list(t.subtrees())):
#         if sub.height() == 2:
#             if 'PUNCT' in sub.label() or 'PUNCT' in sub[0]:  #
#                 parent = sub.parent()
#                 while parent and len(parent) == 1:
#                     sub = parent
#                     parent = sub.parent()
#                 try:
#                     del t[sub.treeposition()]
#                 except:
#                     print(t)
#                     print(t[sub.treeposition()])
#                     raise
#
#     t = nltk.Tree.convert(t)
#     t.collapse_unary(collapsePOS=True, collapseRoot=True)
#     return t


def calc_measures_at_n(n, gold_spans, pred_spans, correct_spans, word_counts):
    mask = word_counts <= n

    gold_spans_sum = gold_spans[mask].sum()
    pred_spans_sum = pred_spans[mask].sum()
    correct_spans_sum = correct_spans[mask].sum()
    r = correct_spans_sum / gold_spans_sum
    p = correct_spans_sum / pred_spans_sum
    f = 2*p*r / (p+r+1e-6)
    logging.info('Length <= {} Rec {:.04f} Prec {:.04f} F1 {:.04f}'.format(n, r, p, f))


def eval_rvm_et_al(args):

    ctx = multiprocessing.get_context('spawn')
    PROCESS_NUM = 6
    if isinstance(args, list):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gold', '-g', required=True, type=str, help='gold tree fn')
        parser.add_argument('--pred', '-p', required=True, type=str, help='predicted tree fn')
        args = parser.parse_args(args)

        gold_fn = args.gold
        pred_fn = args.pred

        if pred_fn.endswith('.gz'):
            with gzip.open(pred_fn, 'rt') as pfh:
                pred_lines = pfh.readlines()
        else:
            with open(pred_fn) as pfh:
                pred_lines = pfh.readlines()
        new_pred_lines = []
        for line in pred_lines:
            if '#!#!' in line:
                _, t = line.split('#!#!')
            else:
                t = line
            if t == "\n":
                t = "(x x)"
            new_pred_lines.append(t)
        pred_lines = new_pred_lines

        with open(gold_fn) as gfh:
            gold_lines = gfh.readlines()

        with ctx.Pool(PROCESS_NUM) as pool:

            gold_trees = pool.map(nltk.Tree.fromstring, gold_lines)
            pred_trees = pool.map(nltk.Tree.fromstring, pred_lines)
    else:
        gold_trees, pred_trees = args

    assert len(gold_trees) == len(pred_trees), "Number of gold trees: {}; number of predicted trees: {}".format(len(gold_trees),
                                                                                                                len(pred_trees))
    for keep_punc_indicator in range(2):

        if keep_punc_indicator == 0:
            logging.info('>>>>> WITHOUT PUNC <<<<<')
        else:
            logging.info('>>>>> WITH PUNC <<<<<')

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 1')
            do_nothing_indicator = [keep_punc_indicator] * len(gold_trees)

            gold_trees_and_punc_indices = pool.starmap(delete_punc, zip(gold_trees, do_nothing_indicator))

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 2')
            processed_gold_trees, gold_indices_of_punc = zip(*gold_trees_and_punc_indices)
            # print(processed_gold_trees[0])

            pred_trees_and_empty_lists = pool.starmap(delete_punc, zip(pred_trees, do_nothing_indicator, gold_indices_of_punc))
            processed_pred_trees, _ = zip(*pred_trees_and_empty_lists)
        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 3')
            processed_gold_trees, processed_pred_trees = zip(*pool.map(single_fix_terms, zip(processed_gold_trees, processed_pred_trees)))


        with ctx.Pool(PROCESS_NUM) as pool:

            # logging.info('step 4')

            total_gold_spans, total_predicted_spans, total_single_word_sent, correct_spans, matching_gold_labels,\
            matching_pred_labels, word_counts, gold_labeled_counts, matching_labeled_consts \
            , matching_cross_labeled_consts , pred_label_counts = zip(*pool.map(eval, zip(processed_gold_trees, processed_pred_trees)))

        # logging.info('step 5')
        total_gold_spans_sum = sum(total_gold_spans)
        total_predicted_spans_sum = sum(total_predicted_spans)
        total_single_word_sent = sum(total_single_word_sent)
        correct_spans_sum = sum(correct_spans)
        matching_gold_labels = list(chain.from_iterable(matching_gold_labels))
        matching_pred_labels = list(chain.from_iterable(matching_pred_labels))
        total_gold_spans = np.array(total_gold_spans)
        total_predicted_spans = np.array(total_predicted_spans)
        correct_spans = np.array(correct_spans)
        word_counts = np.array(word_counts)

        accu_gold_counts = Counter()
        matching_label_counts = Counter()
        acc_pred_counts = Counter()
        matching_cross = Counter()

        # logging.info('step 6')

        for gold_counter, matching_counter, cross_counter, p_counter in zip(gold_labeled_counts, matching_labeled_consts,
                                                             matching_cross_labeled_consts, pred_label_counts):
            accu_gold_counts.update(gold_counter)
            matching_label_counts.update(matching_counter)
            acc_pred_counts.update(p_counter)
            matching_cross.update(cross_counter)
        # logging.info('step 7')
        r = correct_spans_sum / total_gold_spans_sum
        p = correct_spans_sum / total_predicted_spans_sum
        f = 2 * (p * r / (p + r))

        # logging.info('Total single word sent: {}'.format(total_single_word_sent))

        logging.info('*'*50)
        logging.info('Total Rec {:.04f} Prec {:.04f} F1 {:.04f}'.format(r, p, f))
        # logging.info('step 8')

        # logging.info('total gold spans: {}; correct_spans {}'.format(total_gold_spans_sum, correct_spans_sum))
        # calc_measures_at_n(10, total_gold_spans, total_predicted_spans, correct_spans, word_counts)
        # calc_measures_at_n(20, total_gold_spans, total_predicted_spans, correct_spans, word_counts)
        # calc_measures_at_n(30, total_gold_spans, total_predicted_spans, correct_spans, word_counts)
        # calc_measures_at_n(40, total_gold_spans, total_predicted_spans, correct_spans, word_counts)

        # logging.info('step 9')
        ## RVM: recall+VM
        assert len(matching_pred_labels) == len(matching_gold_labels) and len(matching_gold_labels) == correct_spans.sum()
        # import pickle
        # pickle.dump((matching_pred_labels, matching_gold_labels, correct_spans,
        #              accu_gold_counts, matching_cross, acc_pred_counts), open('matching_labels.pkl', 'wb'))
        # m_g_l: gold constituent label, m_p_l: predicted constituent label (their spans match exactly)
        hom, comp, vm = homogeneity_completeness_v_measure(matching_gold_labels, matching_pred_labels)
        logging.info('Homogeneity: {}'.format(hom))
        logging.info('RH: {}'.format(r*hom))

        # for name, count in accu_gold_counts.most_common(8):
        #     logging.info(name + " " + str(matching_label_counts[name] / (1e-6+count)))

    return p, r, f, hom, r*hom


def sanity_check(gold_pred):

    gt = gold_pred[0]
    pt = gold_pred[1]

    if pt.label() == 'x':
        return 0, 0, 0, 0, [], [], 0, Counter(), Counter(), Counter(), Counter()

    g_spans = []
    p_spans = []
    assert len(gt.leaves()) == len(pt.leaves()), "{}\n {}".format(gt, pt)
    # print(gt)
    for subtree in gt.subtrees(lambda x: x.height() > 2):
        g_spans.append(' '.join(subtree.leaves()))

    for subtree in pt.subtrees(lambda x: x.height() > 2):
        p_spans.append(' '.join(subtree.leaves()))

    ggt = gt.copy(deep=True)
    ppt = pt.copy(deep=True)

    this_total_gold_spans = len(g_spans)
    this_total_predicted_spans = len(p_spans)
    this_correct_spans = 0
    len_gold = len(g_spans)
    if len_gold == 0:
        # print('Sent', index, 'Single word sent!', gt)
        return this_total_gold_spans, this_total_predicted_spans, 1, 0, [], [], 1, Counter(), Counter(), Counter(), Counter()
    len_predicted = len(p_spans)

    # p_spans: concatenation of leaves
    # this_correct_spans could be incorrect (not sensitive to span position)
    # probably won't make a difference, but check if there are no edge cases
    # EVALB can always be used to report scores...
    for span in p_spans:
        if span in g_spans:
            this_correct_spans += 1
            g_span_index = g_spans.index(span)
            del g_spans[g_span_index]

    # replaces leaves with integers for integer span
    i = 0
    for g_leaf, p_leaf in zip(ggt.treepositions(order='leaves'), ppt.treepositions(order='leaves')):
        ggt[g_leaf] = str(i)
        ppt[p_leaf] = str(i)
        i += 1

    gg_int_spans = []
    pp_int_spans = []
    this_correct_int_spans = 0

    for subtree in ggt.subtrees(lambda x: x.height() > 2):
        gg_int_spans.append(' '.join(subtree.leaves()))

    for subtree in ppt.subtrees(lambda x: x.height() > 2):
        pp_int_spans.append(' '.join(subtree.leaves()))

    for span in pp_int_spans:
        if span in gg_int_spans:
            this_correct_int_spans += 1
            gg_int_span_index = gg_int_spans.index(span)
            del gg_int_spans[gg_int_span_index]

    # print(this_correct_spans, this_correct_int_spans)
    if this_correct_spans != this_correct_int_spans:
        new_g_spans = []
        new_p_spans = []
        new_gi_spans = []
        new_pi_spans = []
        for subtree in gt.subtrees(lambda x: x.height() > 2):
            new_g_spans.append(' '.join(subtree.leaves()))
        for subtree in pt.subtrees(lambda x: x.height() > 2):
            new_p_spans.append(' '.join(subtree.leaves()))
        for subtree in ggt.subtrees(lambda x: x.height() > 2):
            new_gi_spans.append(' '.join(subtree.leaves()))
        for subtree in ppt.subtrees(lambda x: x.height() > 2):
            new_pi_spans.append(' '.join(subtree.leaves()))

        print(new_g_spans)
        print(new_p_spans)
        print(new_gi_spans)
        print(new_pi_spans)

        for span in new_p_spans:
            if span in new_g_spans:
                print(span)
                g_span_index = new_g_spans.index(span)
                del new_g_spans[g_span_index]

        for span in new_pi_spans:
            if span in new_gi_spans:
                print(span)
                g_span_index = new_gi_spans.index(span)
                del new_gi_spans[g_span_index]

    return this_correct_spans, this_correct_int_spans


def run_s_check(args):

    ctx = multiprocessing.get_context('spawn')
    PROCESS_NUM = 6
    if isinstance(args, list):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gold', '-g', required=True, type=str, help='gold tree fn')
        parser.add_argument('--pred', '-p', required=True, type=str, help='predicted tree fn')
        args = parser.parse_args(args)

        gold_fn = args.gold
        pred_fn = args.pred

        if pred_fn.endswith('.gz'):
            with gzip.open(pred_fn, 'rt') as pfh:
                pred_lines = pfh.readlines()
        else:
            with open(pred_fn) as pfh:
                pred_lines = pfh.readlines()
        new_pred_lines = []
        for line in pred_lines:
            if '#!#!' in line:
                _, t = line.split('#!#!')
            else:
                t = line
            if t == "\n":
                t = "(x x)"
            new_pred_lines.append(t)
        pred_lines = new_pred_lines

        with open(gold_fn) as gfh:
            gold_lines = gfh.readlines()

        with ctx.Pool(PROCESS_NUM) as pool:

            gold_trees = pool.map(nltk.Tree.fromstring, gold_lines)
            pred_trees = pool.map(nltk.Tree.fromstring, pred_lines)
    else:
        gold_trees, pred_trees = args

    print(len(gold_trees), len(pred_trees))
    assert len(gold_trees) == len(pred_trees), "Number of gold trees: {}; number of predicted trees: {}".format(len(gold_trees),
                                                                                                                len(pred_trees))
    for keep_punc_indicator in [1]:

        if keep_punc_indicator == 0:
            logging.info('>>>>> WITHOUT PUNC <<<<<')
        else:
            logging.info('>>>>> WITH PUNC <<<<<')

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 1')
            do_nothing_indicator = [keep_punc_indicator] * len(gold_trees)

            gold_trees_and_punc_indices = pool.starmap(delete_punc, zip(gold_trees, do_nothing_indicator))

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 2')
            processed_gold_trees, gold_indices_of_punc = zip(*gold_trees_and_punc_indices)
            # print(processed_gold_trees[0])

            pred_trees_and_empty_lists = pool.starmap(delete_punc, zip(pred_trees, do_nothing_indicator, gold_indices_of_punc))
            processed_pred_trees, _ = zip(*pred_trees_and_empty_lists)
        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 3')
            processed_gold_trees, processed_pred_trees = zip(*pool.map(single_fix_terms, zip(processed_gold_trees, processed_pred_trees)))


        with ctx.Pool(PROCESS_NUM) as pool:

            # logging.info('step 4')

            correct_spans, correct_int_spans = zip(*pool.map(sanity_check, zip(processed_gold_trees, processed_pred_trees)))

        # correct_spans, correct_int_spans = map(sanity_check, zip(processed_gold_trees, processed_pred_trees))

    # for cspan, cispan in zip(correct_spans, correct_int_spans):
    #     print(cspan, cispan)
    #     if cspan != cispan:
    #         print("WARNING!")

    return True


def w_analysis(gold_pred):

    gt = gold_pred[0]
    pt = gold_pred[1]

    if pt.label() == 'x':
        return {}, {}

    assert len(gt.leaves()) == len(pt.leaves()), "{}\n {}".format(gt, pt)

    plabel_to_glabel = {}
    plabel_to_word = {}

    for pnode, gnode in zip(pt.pos(), gt.pos()):
        plabel_to_glabel[pnode[1]] = plabel_to_glabel.get(pnode[1], [])
        # plabel_to_glabel[pnode[1]].append(gnode[1].split("+")[-1].split("-")[0])
        # collapse_unary results in categories like "S+NP+NNC"
        plabel_to_glabel[pnode[1]].append(gnode[1].split("+")[-1])
        plabel_to_word[pnode[1]] = plabel_to_word.get(pnode[1], [])
        plabel_to_word[pnode[1]].append(pnode[0])

    return plabel_to_glabel, plabel_to_word


def run_w_analysis(args):

    ctx = multiprocessing.get_context('spawn')
    PROCESS_NUM = 6
    if isinstance(args, list):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gold', '-g', required=True, type=str, help='gold tree fn')
        parser.add_argument('--pred', '-p', required=True, type=str, help='predicted tree fn')
        args = parser.parse_args(args)

        gold_fn = args.gold
        pred_fn = args.pred

        if pred_fn.endswith('.gz'):
            with gzip.open(pred_fn, 'rt') as pfh:
                pred_lines = pfh.readlines()
                # print(len(pred_lines))
        else:
            with open(pred_fn) as pfh:
                pred_lines = pfh.readlines()
        new_pred_lines = []
        for line in pred_lines:
            if '#!#!' in line:
                _, t = line.split('#!#!')
            else:
                t = line
            if t == "\n":
                t = "(x x)"
            new_pred_lines.append(t)
        pred_lines = new_pred_lines

        with open(gold_fn) as gfh:
            gold_lines = gfh.readlines()
            # print(len(gold_lines))

        with ctx.Pool(PROCESS_NUM) as pool:

            gold_trees = pool.map(nltk.Tree.fromstring, gold_lines)
            pred_trees = pool.map(nltk.Tree.fromstring, pred_lines)
    else:
        gold_trees, pred_trees = args

    assert len(gold_trees) == len(pred_trees), "Number of gold trees: {}; number of predicted trees: {}".format(len(gold_trees),
                                                                                                                len(pred_trees))
    for keep_punc_indicator in [1]:

        if keep_punc_indicator == 0:
            logging.info('>>>>> WITHOUT PUNC <<<<<')
        else:
            logging.info('>>>>> WITH PUNC <<<<<')

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 1')
            do_nothing_indicator = [keep_punc_indicator] * len(gold_trees)

            gold_trees_and_punc_indices = pool.starmap(delete_punc, zip(gold_trees, do_nothing_indicator))

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 2')
            processed_gold_trees, gold_indices_of_punc = zip(*gold_trees_and_punc_indices)
            # print(processed_gold_trees[0])

            pred_trees_and_empty_lists = pool.starmap(delete_punc, zip(pred_trees, do_nothing_indicator, gold_indices_of_punc))
            processed_pred_trees, _ = zip(*pred_trees_and_empty_lists)

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 3')
            processed_gold_trees, processed_pred_trees = zip(*pool.map(single_fix_terms, zip(processed_gold_trees, processed_pred_trees)))


        with ctx.Pool(PROCESS_NUM) as pool:

            # logging.info('step 4')

            plabel_to_glabel_d, plabel_to_word_d = zip(*pool.map(w_analysis, zip(processed_gold_trees, processed_pred_trees)))

    ptg_accu_d = {}
    ptw_accu_d = {}

    for label_dict, word_dict in zip(plabel_to_glabel_d, plabel_to_word_d):
        for key in label_dict:
            ptg_accu_d[key] = ptg_accu_d.get(key, Counter())
            ptg_accu_d[key].update(label_dict[key])
            ptw_accu_d[key] = ptw_accu_d.get(key, Counter())
            ptw_accu_d[key].update(word_dict[key])

    rank = 1
    for key in sorted(ptg_accu_d, key=lambda x: sum(ptg_accu_d[x].values()),reverse=True):
        print(str(rank) + ". & " + key + " & " + str(sum(ptg_accu_d[key].values())) + " & " + ', '.join(
             [(value + " (" + str(round(count / sum(ptg_accu_d[key].values()), 2)))+")" for value, count in
              ptg_accu_d[key].most_common(10) if (count / sum(ptg_accu_d[key].values())) >= 0.05]) + " \\\\")
        rank += 1
        # print(key, sum(ptg_accu_d[key].values()), [(value, round(count/sum(ptg_accu_d[key].values()), 2)) for value, count in ptg_accu_d[key].most_common(10)])
        # print(key, sum(ptw_accu_d[key].values()), [(value, round(count/sum(ptw_accu_d[key].values()), 2)) for value, count in ptw_accu_d[key].most_common(10)])

    import pickle
    pickle.dump((ptg_accu_d, ptw_accu_d), open('r13e16_induced_prtrm_words.pkl', 'wb'))

    return True


def r_analysis(gold_pred):

    gt = gold_pred[0]
    pt = gold_pred[1]

    if pt.label() == 'x':
        return {}

    # TODO: add word spans for LC and RC too
    prule_to_grule = {}
    prule_to_lcrc = {}
    g_spans = []
    g_labels = []

    assert len(gt.leaves()) == len(pt.leaves()), "{}\n {}".format(gt, pt)
    # print(gt)

    # original word sequence
    words = pt.leaves()

    # replaces leaves with integers for integer span
    i = 0
    for g_leaf, p_leaf in zip(gt.treepositions(order='leaves'), pt.treepositions(order='leaves')):
        gt[g_leaf] = str(i)
        pt[p_leaf] = str(i)
        i += 1

    for subtree in gt.subtrees(lambda x: x.height() > 1):
        g_spans.append(' '.join(subtree.leaves()))
        # g_labels.append(subtree.label().split("+")[-1].split("-")[0])
        g_labels.append(subtree.label().split("+")[-1])

    len_gold = len(g_spans)
    if len_gold == 0:
        return {}

    for subtree in pt.subtrees(lambda x: x.height() > 2):
        parent_span = ' '.join(subtree.leaves())
        lc_span = ' '.join(subtree[0].leaves())
        rc_span = ' '.join(subtree[1].leaves())
        if parent_span in g_spans:
            parent_idx = g_spans.index(parent_span)
            parent_g_label = g_labels[parent_idx]
        else:
            parent_g_label = "??"
        if lc_span in g_spans:
            lc_idx = g_spans.index(lc_span)
            lc_g_label = g_labels[lc_idx]
        else:
            lc_g_label = "??"
        if rc_span in g_spans:
            rc_idx = g_spans.index(rc_span)
            rc_g_label = g_labels[rc_idx]
        else:
            rc_g_label = "??"

        prule = "{} -> {} {}".format(subtree.label(), subtree[0].label(), subtree[1].label())
        grule = "{} -> {} {}".format(parent_g_label, lc_g_label, rc_g_label)
        prule_to_grule[prule] = prule_to_grule.get(prule, [])
        prule_to_grule[prule].append(grule)

        lcrc = "{} | {}".format(" ".join(words[int(subtree[0].leaves()[0]):int(subtree[0].leaves()[-1])+1]),
                                " ".join(words[int(subtree[1].leaves()[0]):int(subtree[1].leaves()[-1])+1]))
        prule_to_lcrc[prule] = prule_to_lcrc.get(prule, [])
        prule_to_lcrc[prule].append(lcrc)

    return prule_to_grule, prule_to_lcrc


def run_r_analysis(args):

    ctx = multiprocessing.get_context('spawn')
    PROCESS_NUM = 6
    if isinstance(args, list):
        parser = argparse.ArgumentParser()
        parser.add_argument('--gold', '-g', required=True, type=str, help='gold tree fn')
        parser.add_argument('--pred', '-p', required=True, type=str, help='predicted tree fn')
        args = parser.parse_args(args)

        gold_fn = args.gold
        pred_fn = args.pred

        if pred_fn.endswith('.gz'):
            with gzip.open(pred_fn, 'rt') as pfh:
                pred_lines = pfh.readlines()
        else:
            with open(pred_fn) as pfh:
                pred_lines = pfh.readlines()
        new_pred_lines = []
        for line in pred_lines:
            if '#!#!' in line:
                _, t = line.split('#!#!')
            else:
                t = line
            if t == "\n":
                t = "(x x)"
            new_pred_lines.append(t)
        pred_lines = new_pred_lines

        with open(gold_fn) as gfh:
            gold_lines = gfh.readlines()

        with ctx.Pool(PROCESS_NUM) as pool:

            gold_trees = pool.map(nltk.Tree.fromstring, gold_lines)
            pred_trees = pool.map(nltk.Tree.fromstring, pred_lines)
    else:
        gold_trees, pred_trees = args

    assert len(gold_trees) == len(pred_trees), "Number of gold trees: {}; number of predicted trees: {}".format(len(gold_trees),
                                                                                                                len(pred_trees))
    for keep_punc_indicator in [1]:

        if keep_punc_indicator == 0:
            logging.info('>>>>> WITHOUT PUNC <<<<<')
        else:
            logging.info('>>>>> WITH PUNC <<<<<')

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 1')
            do_nothing_indicator = [keep_punc_indicator] * len(gold_trees)

            gold_trees_and_punc_indices = pool.starmap(delete_punc, zip(gold_trees, do_nothing_indicator))

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 2')
            processed_gold_trees, gold_indices_of_punc = zip(*gold_trees_and_punc_indices)
            # print(processed_gold_trees[0])

            pred_trees_and_empty_lists = pool.starmap(delete_punc, zip(pred_trees, do_nothing_indicator, gold_indices_of_punc))
            processed_pred_trees, _ = zip(*pred_trees_and_empty_lists)

        with ctx.Pool(PROCESS_NUM) as pool:
            # logging.info('step 3')
            processed_gold_trees, processed_pred_trees = zip(*pool.map(single_fix_terms, zip(processed_gold_trees, processed_pred_trees)))


        with ctx.Pool(PROCESS_NUM) as pool:

            # logging.info('step 4')

            prule_to_grule_d, prule_to_lcrc_d = zip(*pool.map(r_analysis, zip(processed_gold_trees, processed_pred_trees)))

    ptg_accu_d = {}
    ptlr_accu_d = {}

    # print(prule_to_grule_d)

    for rule_dict, span_dict in zip(prule_to_grule_d, prule_to_lcrc_d):
        # print(rule_dict)
        # assert 0 == 1
        for key in rule_dict:
            ptg_accu_d[key] = ptg_accu_d.get(key, Counter())
            ptg_accu_d[key].update(rule_dict[key])
            ptlr_accu_d[key] = ptlr_accu_d.get(key, Counter())
            # print(span_dict[key])
            ptlr_accu_d[key].update(span_dict[key])

    rank = 1
    for key in sorted(ptg_accu_d, key=lambda x: sum(ptg_accu_d[x].values()), reverse=True):
        print(str(rank) + ". & \\makecell{" + key + " \\\\ (" + str(sum(ptg_accu_d[key].values())) + ")} & " + ', '.join(
            [(value + " (" + str(round(count / sum(ptg_accu_d[key].values()), 2))) + ")" for value, count in
             ptg_accu_d[key].most_common(10) if (count / sum(ptg_accu_d[key].values())) >= 0.05]) + " \\\\")
        rank += 1
        # print(key, sum(ptg_accu_d[key].values()), [(value, round(count/sum(ptg_accu_d[key].values()), 2)) for value, count in ptg_accu_d[key].most_common(10)])
        # print(key, sum(ptlr_accu_d[key].values()), [(value, round(count/sum(ptlr_accu_d[key].values()), 2)) for value, count in ptlr_accu_d[key].most_common(10)])

    import pickle
    pickle.dump((ptg_accu_d, ptlr_accu_d), open('r13e16_induced_rules.pkl', 'wb'))

    return True