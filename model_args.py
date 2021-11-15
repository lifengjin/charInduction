import argparse
import os, shutil, random, time
def parse_args(args):
    cmd = argparse.ArgumentParser(args[0], conflict_handler='resolve')
    cmd.add_argument('--seed', default=-1, type=int, help='The random seed.')
    cmd.add_argument('--device', default='cpu', type=str, help='Use id of gpu, -1 if cpu.')

    cmd.add_argument('--train_path', required=True, help='The path to the training file.')
    cmd.add_argument('--train_gold_path', help='The path to the training file with gold trees.')
    cmd.add_argument('--train_markers_path', help='The path to the training file separation markers.')
    cmd.add_argument('--valid_path', help='The path to the development file.')
    cmd.add_argument('--valid_gold_path', help='The path to the development linetrees file.')
    cmd.add_argument('--test_path', help='The path to the testing file.')

    cmd.add_argument('--augment_path', help='the path to the augmentation file', default=None)
    cmd.add_argument('--augment_target', help='augmenting to the number of sentences in train', default=int(5e4), type=int)

    cmd.add_argument("--word_embedding", help="The path to word vectors.")

    cmd.add_argument("--num_nonterminal", default=30, type=int, help="number of nonterminal categories")
    cmd.add_argument("--num_preterminal", default=60, type=int, help="number of preterminal categories")
    cmd.add_argument("--turn_off_char", default=False, action="store_true", help='turn off the char embedder')

    cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam', 'adagrad', 'asgd'],
                     help='the type of optimizer: valid options=[sgd, adam, adagrad, asgd]')
    cmd.add_argument('--max_grad_norm', default=5, type=float, help='gradient clipping parameter')

    cmd.add_argument('--batchnorm', default=False, action='store_true', help='batchnorm for the encoder output')
    cmd.add_argument("--lr", type=float, default=0.001, help='the learning rate.')
    cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
    cmd.add_argument('--category_probability_threshold', type=float, default=5, help='the distance penalty. set to -1 for off, other values for weight.')

    cmd.add_argument("--model", required=True, help="path prefix to save model")
    cmd.add_argument("--model_path", default=None, help="actual path to save model")

    cmd.add_argument("--mid_stage_epoch", default=-1, type=int, help='a staged training schema')
    cmd.add_argument("--forbid_training_pcfg_before_mid_stage", default=False, action='store_true', help="do not set weight to zero before mid stage")

    cmd.add_argument("--batch_size", "--batch", type=int, default=16, help='the batch size.')
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')

    cmd.add_argument("--layer_for_pcfg", type=str, default='0', help='the layer output used for pcfg. set to 0 for embeddings from token encoders')
    cmd.add_argument("--secondary_dropout_rate", type=float, default=0.5, help='the secondary dropout for embeddings')

    cmd.add_argument('--min_count', type=int, default=1, help='minimum word count.')

    cmd.add_argument('--max_vocab_size', type=int, default=150000, help='maximum vocabulary size.')

    cmd.add_argument('--structure_loss_weight', type=float, default=0.1, help='weight of structure loss')

    cmd.add_argument('--valid_size', type=int, default=2000, help="size of validation dataset when there's no valid.")

    cmd.add_argument('--eval_steps', required=False, type=int, default=2, help='report every xx epochs.')
    cmd.add_argument('--eval_train', default=False, action='store_true', help='eval on the training set at the set intervals?')
    cmd.add_argument('--eval_start_epoch', required=False, type=int, default=0, help='the first epoch to start evaling.')

    cmd.add_argument('--logfile', default='log.txt.gz')

    cmd.add_argument('--notes', default='', help='notes for this run')
    cmd.add_argument('--model_type', type=str, default='char', help="'char' for NeuralChar or 'word' for NeuralWord")

    cmd.add_argument("--subgram_word", default=False, action='store_true', help='give subgram features to each word')
    cmd.add_argument("--subgram_stem", default=False, action='store_true', help='give subgram features to each word minus suffix or prefix')

    cmd.add_argument('--char_rnn_type', type=str, default='unified', help='unified or specific')

    cmd.add_argument('--rnn_hidden_dim', type=int, default=512, help='unified or specific')

    cmd.add_argument('--state_dim', type=int, default=128, help='state embedding size')
    cmd.add_argument('--eval_parsing', default=False, action='store_true', help='eval on the training set at the set intervals?')
    cmd.add_argument('--eval_patient', default=5, help='eval on the training set at the set intervals?')

    cmd.add_argument("--korean_phonetics", default=False, action="store_true", help='special treatment for korean: bialphabetic')

    cmd.add_argument("--checkpoint", default="", help='model file to continue training')
    cmd.add_argument("--start_epoch", type=int, default=0, help='epoch to start training')
    cmd.add_argument('--eval_device', default='cpu', type=str, help='Evaluate on CPU or GPU')


    opt = cmd.parse_args(args[2:])
    if opt.model_path is None:
        opt.model_path = os.path.join('outputs', opt.model)
        time.sleep(random.uniform(0, 5))
        for i in range(100):
            checking_path = opt.model_path+'_'+str(i)
            if not os.path.exists(checking_path):
                opt.model_path = checking_path
                break
    else:
        if os.path.exists(opt.model_path):
            shutil.rmtree(opt.model_path)
    os.makedirs(opt.model_path)

    arg_file = os.path.join(opt.model_path, 'args.txt')
    with open(arg_file, 'w') as afh:
        print(vars(opt), file=afh)

    return opt