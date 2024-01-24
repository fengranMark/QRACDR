import torch
from torch.utils.data import Dataset
import json
from tqdm import tqdm, trange
import csv
import random

def padding_seq_to_same_length(input_ids, max_pad_length, pad_token = 0):
    padding_length = max_pad_length - len(input_ids)
    padding_ids = [pad_token] * padding_length
    attention_mask = []

    if padding_length <= 0:
        attention_mask = [1] * max_pad_length
        input_ids = input_ids[:max_pad_length]
    else:
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        input_ids = input_ids + padding_ids
            
    assert len(input_ids) == max_pad_length
    assert len(attention_mask) == max_pad_length
  
    return input_ids, attention_mask
  
class INACDR_qrecc(Dataset):
    def __init__(self, args, tokenizer, filename, rewrite_file=None):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)
        for line in tqdm(data):
            record = json.loads(line)
            flat_concat = []
            cur_utt_text = record['cur_utt_text']
            ctx_utts_text = record['ctx_utts_text']
            cur_response_text = record["cur_response_text"]
            oracle_utt_text = record["oracle_utt_text"]
            
            if "pos_docs_text" in record: #and "random_neg_docs_text" in record:
                pos_docs_text = record["pos_docs_text"]
                if len(pos_docs_text) == 0:
                    continue
                #random_neg_docs_text = record["random_neg_docs_text"]
                if 'bm25_hard_neg_docs' in record:
                    bm25_hard_neg_docs = record['bm25_hard_neg_docs']
            else:
                continue
            
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            pos_docs, pos_docs_mask, neg_docs, neg_docs_mask, = [], [], [], []
            if args.collate_fn_type == "flat_concat_for_train":
                oracle_utt = tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)
                oracle_utt, oracle_utt_mask = padding_seq_to_same_length(oracle_utt, max_pad_length = args.max_query_length)
                for idx in range(len(pos_docs_text)):
                    pos_docs, pos_docs_mask, neg_docs, neg_docs_mask = [], [], [], []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    neg_docs.extend(tokenizer.encode(random.choice(bm25_hard_neg_docs), add_special_tokens=True, max_length = args.max_doc_length))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                    neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                    self.examples.append([record['sample_id'], 
                                    flat_concat,
                                    flat_concat_mask,
                                    pos_docs,
                                    pos_docs_mask,
                                    neg_docs,
                                    neg_docs_mask,
                                    oracle_utt, 
                                    oracle_utt_mask
                                    ])
            
            else:
                oracle_utt, oracle_utt_mask = [], []
                pos_docs, pos_docs_mask, neg_docs, neg_docs_mask = [], [], [], []
                self.examples.append([record['sample_id'], 
                                    flat_concat,
                                    flat_concat_mask,
                                    pos_docs,
                                    pos_docs_mask,
                                    neg_docs,
                                    neg_docs_mask,
                                    oracle_utt, 
                                    oracle_utt_mask
                                    ])

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                             "bt_oracle_labels": [],
                             "bt_oracle_labels_mask": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qa"].append(example[1])
                collated_dict["bt_conv_qa_mask"].append(example[2])
                collated_dict["bt_pos_docs"].append(example[3])
                collated_dict["bt_pos_docs_mask"].append(example[4])
                collated_dict["bt_neg_docs"].append(example[5])
                collated_dict["bt_neg_docs_mask"].append(example[6])
                collated_dict["bt_oracle_labels"].append(example[7])
                collated_dict["bt_oracle_labels_mask"].append(example[8])

            not_need_to_tensor_keys = set(["bt_sample_ids"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
                    
            return collated_dict

        return collate_fn

class INACDR_topiocqa(Dataset):
    def __init__(self, args, tokenizer, filename, rewrite_file):
        self.examples = []
        
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
        with open(rewrite_file, encoding="utf-8") as f:
            rewrite = f.readlines()
        n = len(data)
        n = int(args.use_data_percent * n)  
        # randomly sample n samples for deugging
        if n < len(data):
           random.seed(args.seed)
           data = random.sample(data, n)
        for i in trange(len(data)):
            record = json.loads(data[i])
            flat_concat = []
            ctx_utts_text = record['cur_utt_text'].strip().split(" [SEP] ") # [q1, a1, q2, a2, ...]
            cur_utt_text = ctx_utts_text[-1] 
            ctx_utts_text = ctx_utts_text[:-1]
            oracle_utt_text = json.loads(rewrite[i])["oracle_utt_text"]
            #oracle_utt_text = record["oracle_utt_text"]
            
            pos_docs_text = record["pos_docs"]
            if 'bm25_hard_neg_docs' in record:
                bm25_hard_neg_docs = record['bm25_hard_neg_docs']
            else:
                random_neg_docs_text = record["random_neg_docs_text"]
                
            cur_utt = tokenizer.encode(cur_utt_text, add_special_tokens = True, max_length = args.max_query_length)
            flat_concat.extend(cur_utt)
            for j in range(len(ctx_utts_text) - 1, -1, -1):
                if j % 2 == 1:
                    max_length = args.max_response_length
                else:
                    max_length = args.max_query_length
                utt = tokenizer.encode(ctx_utts_text[j], add_special_tokens=True, max_length=max_length, truncation=True) # not remove [CLS]
                if len(flat_concat) + len(utt) > args.max_concat_length:
                    flat_concat += utt[:args.max_concat_length - len(flat_concat) - 1] + [utt[-1]]    # must ended with [SEP]
                    break
                else:
                    flat_concat.extend(utt) 

            flat_concat, flat_concat_mask = padding_seq_to_same_length(flat_concat, max_pad_length = args.max_concat_length)
            pos_docs, pos_docs_mask, neg_docs, neg_docs_mask = [], [], [], []
            if args.collate_fn_type == "flat_concat_for_train":
                oracle_utt = tokenizer.encode(oracle_utt_text, add_special_tokens = True, max_length = args.max_query_length)
                oracle_utt, oracle_utt_mask = padding_seq_to_same_length(oracle_utt, max_pad_length = args.max_query_length)
                for idx in range(len(pos_docs_text)):
                    pos_docs, pos_docs_mask, neg_docs, neg_docs_mask = [], [], [], []
                    pos_docs.extend(tokenizer.encode(pos_docs_text[idx], add_special_tokens=True, max_length = args.max_doc_length))
                    neg_docs.extend(tokenizer.encode(random.choice(bm25_hard_neg_docs), add_special_tokens=True, max_length = args.max_doc_length))
                    pos_docs, pos_docs_mask = padding_seq_to_same_length(pos_docs, max_pad_length = args.max_doc_length)
                    neg_docs, neg_docs_mask = padding_seq_to_same_length(neg_docs, max_pad_length = args.max_doc_length)
                    self.examples.append([record['sample_id'], 
                                    flat_concat,
                                    flat_concat_mask,
                                    pos_docs,
                                    pos_docs_mask,
                                    neg_docs,
                                    neg_docs_mask,
                                    oracle_utt, 
                                    oracle_utt_mask
                                    ])
            
            else:
                oracle_utt, oracle_utt_mask = [], []
                pos_docs, pos_docs_mask, neg_docs, neg_docs_mask = [], [], [], []
                self.examples.append([record['sample_id'], 
                                    flat_concat,
                                    flat_concat_mask,
                                    pos_docs,
                                    pos_docs_mask,
                                    neg_docs,
                                    neg_docs_mask,
                                    oracle_utt, 
                                    oracle_utt_mask
                                    ])
            

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def get_collate_fn(args):
        
        def collate_fn(batch: list):
            collated_dict = {"bt_sample_ids": [],
                             "bt_conv_qa": [],
                             "bt_conv_qa_mask": [],
                             "bt_pos_docs": [],
                             "bt_pos_docs_mask": [],
                             "bt_neg_docs": [],
                             "bt_neg_docs_mask": [],
                             "bt_oracle_labels": [],
                             "bt_oracle_labels_mask": []
                            }
            for example in batch:
                collated_dict["bt_sample_ids"].append(example[0])
                collated_dict["bt_conv_qa_ids"].append(example[1])
                collated_dict["bt_conv_qa_mask"].append(example[2])
                collated_dict["bt_pos_docs"].append(example[3])
                collated_dict["bt_pos_docs_mask"].append(example[4])
                collated_dict["bt_neg_docs"].append(example[5])
                collated_dict["bt_neg_docs_mask"].append(example[6])
                collated_dict["bt_oracle_labels"].append(example[7])
                collated_dict["bt_oracle_labels_mask"].append(example[8])

            not_need_to_tensor_keys = set(["bt_sample_ids"])

            for key in collated_dict:
                if key not in not_need_to_tensor_keys:
                    collated_dict[key] = torch.tensor(collated_dict[key], dtype=torch.long)
  
            return collated_dict
        return collate_fn
