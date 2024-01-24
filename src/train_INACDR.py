from IPython import embed
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import sys
sys.path.append('..')
sys.path.append('.')

import time
import copy
import pickle
import random
import numpy as np
import csv
import argparse
import toml
import os

from os import path
from os.path import join as oj
import json
from tqdm import tqdm, trange

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaTokenizer
from models import ANCE, ANCE_fuse
#from tensorboardX import SummaryWriter

from utils import check_dir_exist_or_build, pstore, pload, set_seed, get_optimizer, print_res
from data import ConvdrFuse_qrecc, ConvdrFuse_topiocqa

def save_model(args, model, query_tokenizer, save_model_order, epoch, step, loss):
    output_dir = oj(args.model_output_path, '{}-fusetwoCL-best-model'.format("Convdr"))
    check_dir_exist_or_build([output_dir])
    model_to_save = model.module if hasattr(model, 'module') else model
    #model_to_save.t5.save_pretrained(output_dir)
    model_to_save.save_pretrained(output_dir)
    query_tokenizer.save_pretrained(output_dir)
    logger.info("Step {}, Save checkpoint at {}".format(step, output_dir))

def cal_ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_ranking_oracle_loss(query_embs, pos_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    score_mat = pos_scores
    #neg_scores = torch.sum(query_embs * neg_doc_embs, dim = 1).unsqueeze(1) # B * 1 hard negatives
    #score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + 1)  in_batch negatives + 1 BM25 hard negative 
    label_mat = torch.arange(batch_size).to(args.device) # B
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss

def cal_kd_loss(query_embs, kd_embs):
    loss_func = nn.MSELoss()
    return loss_func(query_embs, kd_embs)

def cal_mse_loss_terms(query_embs, doc_embs):
    batch_size = query_embs.size(0)
    embedding_dim = query_embs.size(1)

    mse_loss_func = nn.MSELoss()
    mse_loss = mse_loss_func(query_embs, doc_embs)

    query_embs_l2_norm = torch.linalg.norm(query_embs, ord=2, dim=1)
    doc_embs_l2_norm = torch.linalg.norm(doc_embs, ord=2, dim=1)
    norm_squared_sum = torch.square(query_embs_l2_norm) + torch.square(doc_embs_l2_norm)
    regularization_term = torch.mean(norm_squared_sum / embedding_dim, dim=0)

    dot_product = torch.einsum('ij, ij -> i', query_embs, doc_embs)
    negative_dot_product_term = torch.mean(- 2 / embedding_dim * dot_product, dim=0)
    #assert mse_loss == regularization_term + negative_dot_product_term
    return regularization_term, negative_dot_product_term


def train(args):
    config = RobertaConfig.from_pretrained(args.pretrained_query_encoder_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_query_encoder_path, do_lower_case=True)
    query_encoder = ANCE.from_pretrained(args.pretrained_query_encoder_path, config=config).to(args.device)
    oracle_query_encoder = ANCE.from_pretrained(args.pretrained_oracle_encoder_path, config=config).to(args.device)

    if args.n_gpu > 1:
        query_encoder = torch.nn.DataParallel(query_encoder, device_ids = list(range(args.n_gpu)))

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    # data prepare
    if args.dataset == "topiocqa":
        train_dataset = ConvdrFuse_topiocqa(args, tokenizer, args.train_file_path, args.rewrite_file_path)
    elif args.dataset == "qrecc":
        train_dataset = ConvdrFuse_qrecc(args, tokenizer, args.train_file_path, args.rewrite_file_path)
    train_loader = DataLoader(train_dataset, 
                                #sampler=train_sampler,
                                batch_size = args.batch_size, 
                                shuffle=True, 
                                collate_fn=train_dataset.get_collate_fn(args))

    logger.info("train samples num = {}".format(len(train_dataset)))
    
    total_training_steps = args.num_train_epochs * (len(train_dataset) // args.batch_size + int(bool(len(train_dataset) % args.batch_size)))
    num_warmup_steps = args.num_warmup_portion * total_training_steps
    
    optimizer = get_optimizer(args, query_encoder, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps)

    global_step = 0
    save_model_order = 0

    # begin to train
    logger.info("Start training...")
    logger.info("Total training epochs = {}".format(args.num_train_epochs))
    logger.info("Total training steps = {}".format(total_training_steps))
    
    num_steps_per_epoch = total_training_steps // args.num_train_epochs
    logger.info("Num steps per epoch = {}".format(num_steps_per_epoch))

    if isinstance(args.print_steps, float):
        args.print_steps = int(args.print_steps * num_steps_per_epoch)
        args.print_steps = max(1, args.print_steps)

    epoch_iterator = trange(args.num_train_epochs, desc="Epoch", disable=args.disable_tqdm)

    best_loss = 1000
    for epoch in epoch_iterator:
        query_encoder.train()
        oracle_query_encoder.eval()
        for batch in tqdm(train_loader,  desc="Step", disable=args.disable_tqdm):
            query_encoder.zero_grad()
            bt_conv_query = batch['bt_input_ids'].to(args.device) # B * len
            bt_conv_query_mask = batch['bt_attention_mask'].to(args.device)
            bt_oracle_query = batch['bt_oracle_labels'].to(args.device)
            bt_oracle_query_mask = batch['bt_oracle_labels_mask'].to(args.device)
            bt_pos_docs = batch['bt_pos_docs'].to(args.device) # B * len one pos
            bt_pos_docs_mask = batch['bt_pos_docs_mask'].to(args.device)
            bt_neg_docs = batch['bt_neg_docs'].to(args.device) # B * len one pos
            bt_neg_docs_mask = batch['bt_neg_docs_mask'].to(args.device)

            conv_query_embs = query_encoder(bt_conv_query, bt_conv_query_mask)  # B * dim
            with torch.no_grad():
                # freeze oracle query encoder's parameters
                pos_doc_embs = oracle_query_encoder(bt_pos_docs, bt_pos_docs_mask).detach()  # B * dim
                neg_doc_embs = oracle_query_encoder(bt_neg_docs, bt_neg_docs_mask).detach()  # B * dim
                oracle_utt_embs = oracle_query_encoder(bt_oracle_query, bt_oracle_query_mask).detach()  # B * dim


            
            oracle_loss = cal_kd_loss(conv_query_embs, oracle_utt_embs)
            infusion_loss = cal_kd_loss(conv_query_embs, pos_doc_embs)
            regularization_term, negative_dot_product_term = cal_mse_loss_terms(conv_query_embs, pos_doc_embs)
            ranking_loss = 0.0

            if args.mode == "is_mseneg":
                neg_loss = cal_kd_loss(conv_query_embs, neg_doc_embs)
                loss = (oracle_loss + infusion_loss - neg_loss) / 2
            elif args.mode == "add_CL":
                ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
                loss = (oracle_loss + infusion_loss + ranking_loss) / 3
            elif args.mode == "use_all":
                neg_loss = cal_kd_loss(conv_query_embs, neg_doc_embs)
                ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
                loss = (oracle_loss + infusion_loss - neg_loss + ranking_loss) / 3
            elif args.mode == "CL_only":
                ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
                loss = ranking_loss
            elif args.mode == "reg_term":
                ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
                loss = (oracle_loss + regularization_term + ranking_loss) / 3
            elif args.mode == "ndp_term":
                ranking_loss = cal_ranking_loss(conv_query_embs, pos_doc_embs, neg_doc_embs)
                loss = (oracle_loss + negative_dot_product_term + ranking_loss) / 3
            elif args.mode == "base":
                loss = (oracle_loss + infusion_loss) / 2


            loss.backward()
            torch.nn.utils.clip_grad_norm_(query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            if args.print_steps > 0 and global_step % args.print_steps == 0:
                logger.info("Epoch = {}, Global Step = {}, oracle loss = {}, reg loss = {}, ranking loss = {}, total loss = {}".format(
                                epoch + 1,
                                global_step,
                                oracle_loss.item(),
                                regularization_term.item(),
                                ranking_loss.item(),
                                loss.item()))

            #log_writer.add_scalar("ranking_loss", ranking_loss, global_step)
            #log_writer.add_scalar("oracle_loss", oracle_loss, global_step)
            global_step += 1    # avoid saving the model of the first step.
            # save model finally
            if best_loss > loss:
                save_model(args, query_encoder, tokenizer, save_model_order, epoch, global_step, loss.item())
                best_loss = loss
                logger.info("Epoch = {}, Global Step = {}, total loss = {}".format(
                                epoch + 1,
                                global_step,
                                loss.item()))
                
    logger.info("Training finish!")          
         

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_query_encoder_path", type=str, default="checkpoint/ad-hoc-ance-msmarco")
    parser.add_argument("--pretrained_oracle_encoder_path", type=str, default="checkpoint/ad-hoc-ance-msmarco")

    parser.add_argument("--train_file_path", type=str, default="./datasets/topiocqa/train_with_info.json")
    parser.add_argument("--rewrite_file_path", type=str, default="./datasets/topiocqa/QR/train_T5QR.json")
    parser.add_argument('--model_output_path', type=str, default="../output/topiocqa/model")
    parser.add_argument('--log_dir_path', type=str, default="../loss_log")
    parser.add_argument("--collate_fn_type", type=str, default="flat_concat_for_train")


    parser.add_argument("--per_gpu_train_batch_size", type=int, default=32)
    parser.add_argument("--use_data_percent", type=float, default=1)
    
    parser.add_argument("--num_train_epochs", type=int, default=20, help="num_train_epochs")
    parser.add_argument("--max_query_length", type=int, default=32, help="Max single query length")
    parser.add_argument("--max_doc_length", type=int, default=384, help="Max doc length, consistent with \"Dialog inpainter\".")
    parser.add_argument("--max_response_length", type=int, default=64, help="Max response length, 64 for qrecc, 350 for cast20 since we only have one (last) response")
    parser.add_argument("--max_concat_length", type=int, default=512, help="Max concatenation length of the session. 512 for QReCC.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--mode", type=str, default="mse+CL")
    parser.add_argument("--dataset", type=str, default="topiocqa")
 



    parser.add_argument("--print_steps", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--num_warmup_portion", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    args = parser.parse_args()
    #local_rank = args.local_rank
    #args.local_rank = local_rank

    # pytorch parallel gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#, args.local_rank)
    args.device = device
    #dist.init_process_group(backend='nccl', init_method='env://')
    #torch.cuda.set_device(args.local_rank)

    return args


if __name__ == '__main__':
    args = get_args()
    set_seed(args)
    log_writer = SummaryWriter(log_dir = args.log_dir_path)
    train(args)
    log_writer.close()

