import lzma, os, gzip, tarfile, random, string, re
import argparse

from conllu import parse_incr
parser = argparse.ArgumentParser()
parser.add_argument('--tar', required=True)
parser.add_argument('--max-sent-len', type=int, default=40)
parser.add_argument('--max-corpus-len', type=int, default=1e8)
parser.add_argument('--sorted', default=False, action='store_true')
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--filter-keyword', type=str, default='')
parser.add_argument('--seed', default=4, type=int)
args = parser.parse_args()

TERMINAL_PUNCS = ['.', '?', '!', '。', '"', '”', '？', '！']

if args.seed > 0:
    random.seed(args.seed)

identifier = (''.join([random.choice(string.ascii_letters + string.digits) for n in range(4)])).upper()
file_name = "{}_{}_max{}_total{:.0f}k_sorted{}.txt.gz".format(args.lang, identifier, args.max_sent_len, args.max_corpus_len/1000, int(args.sorted))
output_file = os.path.join(os.path.dirname(args.tar), file_name)
print(file_name)

total_sents = 0

if args.sorted:
    all_sents = [[] for i in range(args.max_sent_len)]
else:
    all_sents = []

with tarfile.open(args.tar, 'r') as tfh, gzip.open(output_file, 'wt') as outputfh:

    for member in tfh:
        if member.isfile() and member.name.endswith('xz'):
            xz_file = tfh.extractfile(member)

            if args.filter_keyword and args.filter_keyword not in member.name:
                continue
            else:
                pass

            print('Doing file:', member.name)
            with lzma.open(xz_file, mode='rt') as ffh:
                for tokenlist in parse_incr(ffh):
                    if not (tokenlist[0]['form'][0].isupper() or tokenlist[0]['form'][0].isalpha()):
                        continue
                    if len(tokenlist) > args.max_sent_len:
                        continue
                    if tokenlist[-1]['upostag'] != 'PUNCT':
                        continue
                    if tokenlist[-1]['form'] not in TERMINAL_PUNCS:
                        continue
                    if tokenlist[-1]['form'].endswith(')'):
                        continue
                    if tokenlist[-2]['upostag'] == 'NUM' and tokenlist[-1]['upostag'] == 'PUNCT' and args.lang == 'german':
                        continue
                    if total_sents > args.max_corpus_len:
                        break
                    sent = []
                    for token in tokenlist:
                        sent.append(token['form'])
                    sent = ' '.join(sent)
                    sent = sent.replace('(', ' ( ')
                    sent = sent.replace(')', ' ) ')
                    sent = sent.replace('[', ' [ ')
                    sent = sent.replace(']', ' ] ')
                    sent = sent.replace('<', ' < ')
                    sent = sent.replace('>', ' > ')
                    sent = sent.replace('~', ' ~ ')
                    sent = sent.replace('/', ' / ')
                    sent = sent.replace('《', ' 《 ')
                    sent = sent.replace('》', ' 》 ')
                    sent = re.sub('\s+', ' ', sent)
                    if sent.count(' ') > args.max_sent_len:
                        continue
                    if args.sorted:
                        all_sents[len(tokenlist)].append(sent)
                    else:
                        all_sents.append(sent)
                    total_sents += 1
            if total_sents > args.max_corpus_len:
                break

    if args.sorted:
        for bin in all_sents:
            if bin:
                shuffled_sent_ids = list(range(len(bin)))
                random.shuffle(shuffled_sent_ids)
                for sent_id in shuffled_sent_ids:
                    print(bin[sent_id], file=outputfh)
    else:
        shuffled_sent_ids = list(range(len(all_sents)))
        random.shuffle(shuffled_sent_ids)
        for sent_id in shuffled_sent_ids:
            print(all_sents[sent_id], file=outputfh)