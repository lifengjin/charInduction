# Character-based PCFG Induction for Modeling the Syntactic Acquisition of Morphologically Rich Languages

## Introduction
This is the code repository for the paper [Character-based PCFG Induction for Modeling the Syntactic Acquisition of Morphologically Rich Languages](https://aclanthology.org/2021.findings-emnlp.371/), including unsupervised PCFG induction models as well as manually corrected syntactic annotations for Korean child-directed speech.

Major dependencies include:

- Python 3.7+
- PyTorch 1.7.0+
- TensorBoard 2.3.0+
- Bidict

## Training
`main.py` is the main training script. Sample commands for training the induction model can be found under the `exps` directory:

```
python main.py train --seed -1 \
                     --train_path data/Jong.010322.linetoks \
                     --train_gold_path data/Jong.010322.linetrees \
                     --model_type char --model char_jong_c90 \
                     --num_nonterminal 45 --num_preterminal 45 \
                     --state_dim 128 --rnn_hidden_dim 512 \
                     --max_epoch 45 --batch_size 2 \
                     --optimizer adam --device cuda \
                     --eval_device cuda --eval_steps 2 \
                     --eval_start_epoch 1 --eval_parsing
```

`seed`: Seed for the run, -1 can be used for a random seed.

`train_path`: Path to training sentences, which should be whitespace-tokenized like the following:
```
이리로 와 엄마랑 보자 .
이게 누구에요 ?
아우 이쁘다 우리 종현이네 .
```

`train_gold_path`: Path to gold trees for training sentences, which are used only for evaluation:
```
(S (S (ADVP (npd+jca 이리로)) (VP (pvg+ef 와))) (S (VP (NP (ncn+jcj 엄마랑)) (pvg+ef 보자))) (sf .))
(S (NP (npd+jcs 이게)) (VP (npp+jp+ef 누구에요)) (sf ?))
(S (S (IP (ii 아우)) (VP (paa+ef 이쁘다))) (S (ADJP (npp 우리)) (VP (nq+jp+ef 종현이네)) (sf .)))
```

`model_type`: Use `char` for the NeuralChar model and `word` for the NeuralWord model described in the paper.

`model`: The name of the directory in which the output will be saved saved. For example, if `char_jong_c90` is used, then the run info will be saved under `outputs/char_jong_c90_0` for the first run, `outputs/char_jong_c90_1` for the second run, and so on.

`num_nonterminal` and `num_preterminal`: The NeuralChar and NeuralWord models do not make a distinction between preterminal categories and other nonterminal categories. Set these two values accordingly so that they add up total number of categories to be used (e.g. 45 and 45 for a total of 90 categories).

`state_dim`: Dimensionality of category embeddings.

`rnn_hidden_dim`: Dimensionality of the LSTM hidden state for the NeuralChar model's emission probabilities.

`eval_steps`: Number of training epochs between each evaluation.

`eval_start_epoch`: Number of training epochs before first evaluation.

`eval_parsing`: Whether or not to parse training sentences as part of evaluation. If set to `False`, only the likelihood will be reported.

Descriptions of other arguments can be found in `model_args.py`.

## Syntactic Annotations of Korean Child-Directed Speech
The `data` directory contains manually corrected syntactic annotations of Korean child-directed speech from the [Ryu corpus of the CHILDES database](https://doi.org/10.1111/lang.12132). The annotation scheme follows that of [Choi (2013)](http://arxiv.org/abs/1309.1649).

## Questions
For questions or concerns, please contact Lifeng Jin ([lifengjin@tencent.com](mailto:lifengjin@tencent.com)) or Byung-Doh Oh ([oh.531@osu.edu](mailto:oh.531@osu.edu)).
