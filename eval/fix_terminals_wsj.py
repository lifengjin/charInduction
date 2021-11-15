import nltk
import argparse
import gzip
# this file is for fixing the different terminal symbols caused by tokenization.
# it takes a gold linetree and a predicted linetree file
# and replace the tokens in the predicted linetree file with gold if they are different.

def single_fix_terms(gold_tree_pred_tree):
    this_gold_tree = gold_tree_pred_tree[0]
    this_predicted_tree = gold_tree_pred_tree[1]
    if this_predicted_tree.label() == 'x':
        return this_gold_tree, this_predicted_tree
    gold_tokens = this_gold_tree.leaves()
    predicted_tokens = this_predicted_tree.leaves()
    if len(gold_tokens) != len(predicted_tokens):
        print("gold tokens are {}".format(gold_tokens))
        print("predicted tokens are {}".format(predicted_tokens))
        raise Exception
    for j in range(len(gold_tokens)):
        if gold_tokens[j] != predicted_tokens[j]:
            this_predicted_tree[this_predicted_tree.leaf_treeposition(j)] = gold_tokens[j]
    return this_gold_tree, this_predicted_tree

def fix_terms(gold_trees, predicted_trees):
    for i in range(len(gold_trees)):
        this_gold_tree = gold_trees[i]
        this_predicted_tree = predicted_trees[i]
        single_fix_terms((this_gold_tree, this_predicted_tree))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold', '-g', required=True, help='the gold linetrees file')
    parser.add_argument('--pred', '-p', dest='predicted', required=True, help='the predicted linetrees file')
    args = parser.parse_args()

    gold_trees = []
    with open(args.gold) as g:
        for line in g:
            line = line.strip()
            gold_trees.append(nltk.tree.Tree.fromstring(line))

    predicted_trees = []
    if args.predicted.endswith('gz'):
        with gzip.open(args.predicted, 'rt') as pfh:
            alllines = pfh.readlines()
    else:

        with open(args.predicted) as p:
            alllines = p.readlines()
    for line in alllines:
        line = line.strip()
        predicted_trees.append(nltk.tree.Tree.fromstring(line))

    fix_terms(gold_trees, predicted_trees)

    fn = args.predicted.split('.')

    fn = fn[:-1]
    fn.append('fixterms.linetrees')
    fn = '.'.join(fn)
    with open(fn, 'w') as ft:
        for tree in predicted_trees:
            string = tree.pformat(margin=100000)
            print(string, file=ft)