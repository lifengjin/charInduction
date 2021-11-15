from .compare_trees import main as sent_f1
from .evalb_unlabeled import eval_rvm_et_al
import tempfile
import nltk

def eval_access(pred_tree_list, gold_tree_list, writer, epoch, section='dev'):

    gold_trees = []
    for t in gold_tree_list:
        gold_trees.append(nltk.tree.Tree.fromstring(t))

    p, r, f1, vm, rvm = eval_rvm_et_al((gold_trees, pred_tree_list))

    # writer.add_scalar(section+'_epochwise/corpus_f1', corpus_f1_val, epoch)
    # writer.add_scalar(section+'_epochwise/corpus_recall', corpus_recall_val, epoch)
    # writer.add_scalar(section+'_epochwise/sent_recall', sent_recall_val, epoch)
    # writer.add_scalar(section+'_epochwise/sent_f1', sent_f1_val, epoch)
    if writer is not None:
        writer.add_scalar(section+'_epochwise/p', p, epoch)
        writer.add_scalar(section+'_epochwise/r', r, epoch)
        writer.add_scalar(section+'_epochwise/f1', f1, epoch)
        writer.add_scalar(section+'_epochwise/vm', vm, epoch)
        writer.add_scalar(section+'_epochwise/rvm', rvm, epoch)
