import random
import torch
import time
import logging
import numpy as np

def slice_tensor(tensor, batch_size, flatten=False):
    if tensor is None:
        return tensor
    return list([tensor[i:i+batch_size].flatten() if flatten else tensor[i:i+batch_size]
                 for i in range(0, tensor.shape[0], batch_size)])


def train_model(epoch, opt, model, optimizer, train):
    """
    Training model for one epoch
    """
    model.train()

    slicing_flag = False

    total_loss, total_tag, total_chars, total_acc_chars = 1e-7, 1e-7, 1e-7, 1e-7

    cnt = 0
    start_time = time.time()

    train_w, train_c, train_var_c, train_lens, train_masks, train_indices = train
    max_cnt = len(train_w)
    tenths = list([int(max_cnt / 10) * i for i in range(1, 10)])

    lst = list(range(len(train_w)))

    train_w = [train_w[l] for l in lst]
    train_c = [train_c[l] for l in lst]
    train_var_c = [train_var_c[l] for l in lst]
    train_lens = [train_lens[l] for l in lst]
    train_masks = [train_masks[l] for l in lst]
    train_indices = [train_indices[l] for l in lst]

    for w, c, var_c, lens, masks, indices in zip(train_w, train_c, train_var_c, train_lens, train_masks, train_indices):
        cnt += 1
        optimizer.zero_grad()

        if slicing_flag:
            ws, cs = slice_tensor(w, opt.batch_size), slice_tensor(c, opt.batch_size)
            if cs is None:
                cs = [None] * len(ws)
            sliced_masks = []
            sliced_masks.append(slice_tensor(masks[0], opt.batch_size))
            master_mask = torch.arange(opt.batch_size * w.shape[1], device=w.device).reshape(opt.batch_size, -1)
            fw_mask = master_mask[:, :-1]
            bw_mask = master_mask[:, 1:]
            sliced_masks.append([fw_mask[:wss.shape[0]].flatten() for wss in ws])
            sliced_masks.append([bw_mask[:wss.shape[0]].flatten() for wss in ws])
            sliced_masks = zip(*sliced_masks)

        else:
            ws, cs, var_cs, sliced_masks = (w,), (c,), (var_c,), (masks,)

        for ww, cc, varcc, mm in zip(ws, cs, var_cs, sliced_masks):
            loss = model.forward(ww, varcc)
            loss.backward()
            total_loss += loss.item()

        total_tag += sum(lens)

        if opt.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.max_grad_norm)
        optimizer.step()

        global_step = max_cnt * epoch + cnt

        if cnt in tenths:
            logging.info("Epoch={} iter={} lr={:.5f} train loss={:.4f} time={:.2f}s".format(
                epoch, cnt, optimizer.param_groups[0]['lr'],
                total_loss / total_tag, time.time() - start_time
            ))

            start_time = time.time()
            model.writer.add_scalar('train_accumulative/average_total_loss', total_loss / total_tag, global_step)

    model.writer.add_scalar('train_epochwise/average_total_loss', total_loss / total_tag, epoch)

    return optimizer

def parse_dataset(model, dataset, epoch, section='dev'):
    model.eval()
    with torch.no_grad():
        train_w, train_c, train_var_c, train_lens, train_masks, train_indices = dataset
        trees = [None] * sum([len(x) for x in train_indices])
        total_structure_loss = 0
        total_num_tags = sum([sum(x) for x in train_lens])
        for batch_index, (w, c, var_c, lens, masks, indices) in enumerate(zip(train_w, train_c, train_var_c, train_lens,
                                                                              train_masks, train_indices)):
            if batch_index == 0:
                structure_loss, v_treelist = model.parse(w, var_c, indices, set_pcfg=True)
            else:
                structure_loss, v_treelist = model.parse(w, var_c, indices, set_pcfg=False)

            for t_id, t in zip(indices, v_treelist):
                trees[t_id] = t

            total_structure_loss += structure_loss
        if model.writer is not None:
            model.writer.add_scalar(section+'_epochwise/average_structure_loss', total_structure_loss / total_num_tags, epoch)
            logging.info(
                'Epoch {} EVALUATION | Structure loss {:.4f} '.format(epoch, total_structure_loss))
    model.train()
    return total_structure_loss, trees

def likelihood_dataset(model, dataset, epoch, section='dev'):
    model.eval()
    with torch.no_grad():
        train_w, train_c, train_var_c, train_lens, train_masks, train_indices = dataset
        total_structure_loss = 0
        total_num_tags = sum([sum(x) for x in train_lens])
        for batch_index, (w, c, var_c, lens, masks, indices) in enumerate(zip(train_w, train_c, train_var_c, train_lens,
                                                                              train_masks, train_indices)):
            if batch_index == 0:
                structure_loss = model.likelihood(w, var_c, indices, set_pcfg=True)
            else:
                structure_loss = model.likelihood(w, var_c, indices, set_pcfg=False)

            total_structure_loss += structure_loss

        model.writer.add_scalar(section+'_epochwise/average_structure_loss', total_structure_loss / total_num_tags, epoch)
        logging.info(
            'Epoch {} EVALUATION | Structure loss {:.4f} '.format(epoch, total_structure_loss))
    model.train()
    return total_structure_loss

def eval_model(model, valid):
    model.eval()
    if model.config['classifier']['name'].lower() == 'cnn_softmax' or \
            model.config['classifier']['name'].lower() == 'sampled_softmax':
        model.classify_layer.update_embedding_matrix()
    total_loss, total_tag = 0.0, 0
    valid_w, valid_c, valid_lens, valid_masks = valid
    for w, c, lens, masks in zip(valid_w, valid_c, valid_lens, valid_masks):
        loss_forward, loss_backward = model.forward(w, c, masks)
        total_loss += loss_forward.data[0]
        n_tags = sum(lens)
        total_tag += n_tags
    model.train()
    return np.exp(total_loss / total_tag)