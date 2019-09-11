import argparse
import copy
import json
import pdb
import random
import re
import traceback
from collections import defaultdict, namedtuple
from itertools import product

import numpy as np
import torch
import torch.multiprocessing as mp
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from data import convert_question_to_samples_bundle, homebrew_data_loader
from model import BertForMultiHopQuestionAnswering, CognitiveGNN
from utils import (WindowMean, bundle_part_to_batch,
                   find_start_end_after_tokenized,
                   find_start_end_before_tokenized, fuzz, fuzzy_retrieve,
                   judge_question_type, warmup_linear)


def train(bundles, model1, device, mode, model2, batch_size, num_epoch, gradient_accumulation_steps, lr1, lr2, alpha, expname):
    '''Train Sys1 and Sys2 models.
    
    Train models by task #1(tensors) and task #2(bundle). 
    
    Args:
        bundles (list): List of bundles.
        model1 (BertForMultiHopQuestionAnswering): System 1 model.
        device (torch.device): The device which models and data are on.
        mode (str): Defaults to 'tensors'. Task identifier('tensors' or 'bundle').
        model2 (CognitiveGNN): System 2 model.
        batch_size (int): Defaults to 4.
        num_epoch (int): Defaults to 1.
        gradient_accumulation_steps (int): Defaults to 1. 
        lr1 (float): Defaults to 1e-4. Learning rate for Sys1.
        lr2 (float): Defaults to 1e-4. Learning rate for Sys2.
        alpha (float): Defaults to 0.2. Balance factor for loss of two systems.
    
    Returns:
        ([type], [type]): Trained models.
    '''

    # Prepare optimizer for Sys1
    param_optimizer = list(model1.named_parameters())
    # hack to remove pooler, which is not used.
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_batch, dataloader = homebrew_data_loader(bundles, mode = mode, batch_size=batch_size)
    num_steps = num_batch * num_epoch
    global_step = 0
    opt1 = BertAdam(optimizer_grouped_parameters, lr = lr1, warmup = 0.1, t_total=num_steps)
    model1.to(device)
    model1.train()

    # Prepare optimizer for Sys2
    if mode == 'bundle':
        opt2 = Adam(model2.parameters(), lr=lr2)
        model2.to(device)
        model2.train()
        warmed = False # warmup for jointly training

    writer = SummaryWriter(f"runs/{expname}")
    for epoch in trange(num_epoch, desc = 'Epoch'):
        ans_mean, hop_mean = WindowMean(), WindowMean()
        opt1.zero_grad()
        if mode == 'bundle':
            final_mean = WindowMean()
            opt2.zero_grad()
        tqdm_obj = tqdm(dataloader, total = num_batch)

        losses = defaultdict(list)
        for step, batch in enumerate(tqdm_obj):
            try:
                if mode == 'tensors':
                    batch = tuple(t.to(device) for t in batch)
                    hop_loss, ans_loss, pooled_output = model1(*batch)
                    hop_loss, ans_loss = hop_loss.mean(), ans_loss.mean()
                    pooled_output.detach()
                    loss = ans_loss + hop_loss
                elif mode == 'bundle':
                    hop_loss, ans_loss, final_loss = model2(batch, model1, device)
                    hop_loss, ans_loss = hop_loss.mean(), ans_loss.mean()
                    loss = ans_loss + hop_loss + alpha * final_loss
                    losses['final'].append(final_loss.item())
                losses['total'].append(loss.item())
                losses['hop'].append(hop_loss.item())
                losses['ans'].append(ans_loss.item())
                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses. From BERT pytorch examples
                    # lr_this_step = lr1 * warmup_linear(global_step/num_steps, warmup = 0.1)
                    lr_this_step = opt1.get_lr()
                    assert len(set(lr_this_step)) == 1
                    lr_this_step = lr1 * lr_this_step[0]
                    global_step += 1
                    if mode == 'bundle':
                        opt2.step()
                        opt2.zero_grad()
                        final_mean_loss = final_mean.update(final_loss.item())
                        tqdm_obj.set_description('ans_loss: {:.2f}, hop_loss: {:.2f}, final_loss: {:.2f}'.format(
                            ans_mean.update(ans_loss.item()), hop_mean.update(hop_loss.item()), final_mean_loss))
                        # During warming period, model1 is frozen and model2 is trained to normal weights
                        if final_mean_loss < 0.9 and step > 100: # ugly manual hyperparam
                            warmed = True
                        if warmed:
                            opt1.step()
                        opt1.zero_grad()
                    else:
                        opt1.step()
                        opt1.zero_grad()
                        tqdm_obj.set_description('ans_loss: {:.2f}, hop_loss: {:.2f}'.format(
                            ans_mean.update(ans_loss.item()), hop_mean.update(hop_loss.item())))
                    if step % 1000 == 0:
                        output_model_file = f'./models/bert-base-uncased-{expname}.bin.tmp'
                        saved_dict = {'params1' : model1.module.state_dict()}
                        saved_dict['params2'] = model2.state_dict()
                        torch.save(saved_dict, output_model_file)

                    if (step + 1) % 100 == 0:
                        for k in losses:
                            writer.add_scalar(f'loss/{k}', np.mean(losses[k]), step)
                        losses.clear()
                        writer.add_scalar(f'lr', lr_this_step, step)

            except Exception as err:
                traceback.print_exc()
                if mode == 'bundle':   
                    print(batch._id) 
    return (model1, model2)


def main(output_model_file = './models/bert-base-uncased.bin', load = False, mode = 'tensors', batch_size = 20, 
            num_epoch = 1, gradient_accumulation_steps = 1, lr1 = 1e-4, lr2 = 1e-4, alpha = 0.2, expname=""):
    
    BERT_MODEL = 'bert-base-uncased' # bert-large is too large for ordinary GPU on task #2
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    with open('./hotpot_train_v1.1_refined.json' ,'r') as fin:
        dataset = json.load(fin)
    bundles = []
    for data in tqdm(dataset[:10000]):
        try:
            bundles.append(convert_question_to_samples_bundle(tokenizer, data))
        except ValueError as err:
            pass
        # except Exception as err:
        #     traceback.print_exc()
        #     pass
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    if load:
        print('Loading model from {}'.format(output_model_file))
        model_state_dict = torch.load(output_model_file)
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL, state_dict=model_state_dict['params1'])
        model2 = CognitiveGNN(model1.config.hidden_size, model1.config)
        model2.load_state_dict(model_state_dict['params2'])
        from model import XAttn
        model2.gcn = XAttn(model1.config.hidden_size, model1.config)

    else:
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(BERT_MODEL,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1))
        model2 = CognitiveGNN(model1.config.hidden_size, model1.config)

    print('Start Training... on {} GPUs'.format(torch.cuda.device_count()))
    model1 = torch.nn.DataParallel(model1, device_ids = range(torch.cuda.device_count()))
    model1, model2 = train(bundles, model1=model1, device=device, mode=mode, model2=model2, # Then pass hyperparams
        batch_size=batch_size, num_epoch=num_epoch, gradient_accumulation_steps=gradient_accumulation_steps,lr1=lr1, lr2=lr2, alpha=alpha, expname=expname)
    
    output_model_file = f'./models/bert-base-uncased-{expname}.bin'
    print('Saving model to {}'.format(output_model_file))
    saved_dict = {'params1' : model1.module.state_dict()}
    saved_dict['params2'] = model2.state_dict()
    torch.save(saved_dict, output_model_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str, default="expname", help="Name of the experiment for logging and saving.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--mode", type=str, default="tensors")
    parser.add_argument("--lr1", type=float, default=1e-4)
    args = parser.parse_args()
    main(load=args.load, mode=args.mode, batch_size=args.batch_size, expname=args.expname, lr1=args.lr1)
