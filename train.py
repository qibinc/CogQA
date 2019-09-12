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

from dataset import convert_question_to_samples_bundle, homebrew_data_loader
from model import BertForMultiHopQuestionAnswering, CognitiveGNN
from utils import (bundle_part_to_batch, find_start_end_after_tokenized,
                   find_start_end_before_tokenized, fuzz, fuzzy_retrieve,
                   judge_question_type, warmup_linear)


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--load", action="store_true", default=False)
    parser.add_argument("--load-path", type=str, default="./saved/bert-base-uncased.bin")
    parser.add_argument("--mode", type=str, default="#1")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--lr1", type=float, default=1e-4)
    parser.add_argument("--lr2", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--tune", action="store_true", default=False)
    # bert-large is too large for ordinary GPU on task #2
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased")
    parser.add_argument("--xattn-layers", type=int, default=1)
    parser.add_argument("--sys2", type=str, default="xattn", choices=["xattn", "gcn", "mlp"])
    parser.add_argument(
        "--expname",
        type=str,
        default="expname",
        help="Name of the experiment for logging and saving.",
    )
    # fmt: off
    return parser.parse_args()

def save_model(model1, model2, save_path):
    print(f"Saving model to {save_path}")
    state_dict = {
        "params1": model1.module.state_dict(),
        "params2": model2.state_dict()
    }
    torch.save(state_dict, save_path)

def train(
    train_bundles,
    valid_bundles,
    model1,
    mode,
    model2,
    batch_size,
    num_epochs,
    gradient_accumulation_steps,
    lr1,
    lr2,
    weight_decay,
    expname,
    bert_model,
):
    """Train Sys1 and Sys2 models.
    
    Train models by task #1 and task #2. 
    
    Args:
        bundles (list): List of bundles.
        model1 (BertForMultiHopQuestionAnswering): System 1 model.
        device (torch.device): The device which models and data are on.
        mode (str): Defaults to '#1'. Task identifier('#1' or '#2').
        model2 (CognitiveGNN): System 2 model.
        batch_size (int): Defaults to 4.
        num_epoch (int): Defaults to 1.
        gradient_accumulation_steps (int): Defaults to 1. 
        lr1 (float): Defaults to 1e-4. Learning rate for Sys1.
        lr2 (float): Defaults to 1e-4. Learning rate for Sys2.
    
    Returns:
        ([type], [type]): Trained models.
    """

    print("Start Training... on {} GPUs".format(torch.cuda.device_count()))
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )

    # Prepare optimizer for Sys1
    param_optimizer = list(model1.named_parameters())
    # hack to remove pooler, which is not used.
    param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_batches, dataloader = homebrew_data_loader(
        train_bundles, mode=mode, batch_size=batch_size
    )
    num_steps = num_batches * num_epochs // gradient_accumulation_steps

    opt1 = BertAdam(optimizer_grouped_parameters, lr=lr1, warmup=0.1, t_total=num_steps)
    model1.to(device)
    model1.train()

    if mode == "#2":
        opt2 = Adam(model2.parameters(), lr=lr2)
        model2.to(device)
        model2.train()
        # warmed = False  # warmup for jointly training

    global_step = 0
    writer = SummaryWriter(f"saved/{expname}")
    for _ in trange(num_epochs):
        opt1.zero_grad()
        if mode == "#2":
            opt2.zero_grad()

        losses = defaultdict(list)
        for step, batch in enumerate(tqdm(dataloader, total=num_batches)):
            try:
                if mode == "#1":
                    batch = tuple(t.to(device) for t in batch)
                    hop_loss, ans_loss, _ = model1(*batch)
                    hop_loss, ans_loss = hop_loss.mean(), ans_loss.mean()
                    loss = ans_loss + hop_loss
                    losses["hop"].append(hop_loss.item())
                    losses["ans"].append(ans_loss.item())
                elif mode == "#2":
                    _, _, final_loss = model2(batch, model1, device)
                    loss = final_loss
                    losses["final"].append(final_loss.item())

                losses["total"].append(loss.item())

                if gradient_accumulation_steps > 1:
                    loss /= gradient_accumulation_steps
                loss.backward()

                if (step + 1) % gradient_accumulation_steps != 0:
                    continue

                # modify learning rate with special warm up BERT uses. From BERT pytorch examples
                # lr_this_step = lr1 * warmup_linear(global_step/num_steps, warmup = 0.1)
                # lr_this_step = opt1.get_lr()
                # assert len(set(lr_this_step)) == 1
                # lr_this_step = lr1 * lr_this_step[0]
                global_step += 1
                if mode == "#2":
                    opt2.step()
                    opt2.zero_grad()
                else:
                    opt1.step()
                    opt1.zero_grad()

                if (step + 1) % 1000 == 0:
                    save_path = (
                        f"./saved/{bert_model}-{expname}.bin.tmp"
                    )
                    save_model(model1, model2, save_path)

                if (step + 1) % 100 == 0:
                    for k in losses:
                        writer.add_scalar(f"loss/{k}", np.mean(losses[k]), step)
                    losses.clear()
                    # writer.add_scalar(f"lr", lr_this_step, step)

            except Exception as err:
                traceback.print_exc()

    return model1, model2


if __name__ == "__main__":
    args = parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    with open("./data/hotpot_train_v1.1_refined.json", "r") as fin:
        train_data = json.load(fin)
    with open("./data/hotpot_dev_distractor_v1_refined.json", "r") as fin:
        valid_data = json.load(fin)
    train_bundles = []
    # Use a portion of dataset for tuning
    if args.tune:
        train_data = train_data[:10000]
    for data in tqdm(train_data):
        try:
            train_bundles.append(convert_question_to_samples_bundle(tokenizer, data))
        except ValueError as err:
            pass
    valid_bundles = []
    # for data in tqdm(valid_data):
    #     try:
    #         valid_bundles.append(convert_question_to_samples_bundle(tokenizer, data))
    #     except ValueError as err:
    #         pass

    if not args.load:
        # Task #1
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(
            args.bert_model,
            cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / "distributed_{}".format(-1),
        )
        model2 = CognitiveGNN(model1.config.hidden_size, model1.config, args.sys2)
    else:
        # Task #2
        print("Loading model from {}".format(args.load_path))
        model_state_dict = torch.load(args.load_path)
        model1 = BertForMultiHopQuestionAnswering.from_pretrained(
            args.bert_model, state_dict=model_state_dict["params1"]
        )
        hidden_size = model1.config.hidden_size
        model2 = CognitiveGNN(hidden_size, model1.config, args.sys2)
        model2.load_state_dict(model_state_dict["params2"])
        if args.sys2 == "xattn":
            from model import XAttn
            model2.gcn = XAttn(model1.config.hidden_size, model1.config, n_layers=args.xattn_layers)
        elif args.sys2 == "mlp":
            from layers import MLP
            model2.gcn = MLP((hidden_size, hidden_size, 1))

    model1 = torch.nn.DataParallel(model1, device_ids=range(torch.cuda.device_count()))
    model1, model2 = train(
        train_bundles,
        valid_bundles,
        model1=model1,
        mode=args.mode,
        model2=model2,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr1=args.lr1,
        lr2=args.lr2,
        weight_decay=args.weight_decay,
        expname=args.expname,
        bert_model=args.bert_model
    )

    save_path = f"./saved/{args.bert_model}-{args.expname}.bin"
    save_model(model1, model2, save_path)
