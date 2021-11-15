python main.py train --seed -1 \
                     --train_path data/Jong.010322.linetoks \
                     --train_gold_path data/Jong.010322.linetrees \
                     --model_type word --model word_jong_c90 \
                     --num_nonterminal 45 --num_preterminal 45 \
                     --state_dim 128 --rnn_hidden_dim 512 \
                     --max_epoch 45 --batch_size 2 \
                     --optimizer adam --device cuda \
                     --eval_device cuda --eval_steps 2 \
                     --eval_start_epoch 1 --eval_parsing
