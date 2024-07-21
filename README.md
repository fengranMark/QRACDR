# QRACDR
This is the temporary repository of our CIKM 2024 accepted paper - Aligning Query Representation with Rewritten Query and Relevance Judgments in Conversational Search.

# Environment Dependency

Main packages:
- python 3.8
- torch 1.8.1
- transformer 4.2.0
- numpy 1.22
- faiss-gpu 1.7.2

# Running Steps

## 1. Download data and Preprocessing

Conversational search datasets can be downloaded from [QReCC](https://github.com/apple/ml-qrecc), [TopiOCQA](https://github.com/McGill-NLP/topiocqa), and [TREC-CAST](https://www.treccast.ai/). Then run the scripts in the preprocess folder for data preprocessing.

## 2. Retrieval Indexing

To evaluate the trained model by QRACDR, we should first establish index. We use the pre-trained ad-hoc search model ANCE to generate document embeddings. Two scripts for each dataset are provided in index folder by running:

    python gen_tokenized_doc.py --config=gen_tokenized_doc.toml
    python gen_doc_embeddings.py --config=gen_doc_embeddings.toml

## 3. Train QRACDR

To train QRACDR, please run the following commands in the src folder. The pre-trained language model we use for dense retrieval is [ANCE](https://github.com/microsoft/ANCE).

    python train_QRACDR.py --pretrained_encoder_path="checkpoints/ad-hoc-ance-msmarco" \ 
      --train_file_path=$train_file_path \ 
      --log_dir_path=$log_dir_path \
      --model_output_path=$model_output_path \ 
      --per_gpu_train_batch_size=32 \ 
      --num_train_epochs=10 \
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=64 \
      --max_concat_length=512 \ 
      --dataset="topiocqa" \
      --mode="mse+CL" \

## 4. Retrieval evaluation

Now, we can perform retrieval to evaluate the QRACDR-trained conversational dense retriever by running:

    python test_QRACDR.py --pretrained_encoder_path=$trained_model_path \ 
      --passage_embeddings_dir_path=$passage_embeddings_dir_path \ 
      --passage_offset2pid_path=$passage_offset2pid_path \
      --qrel_output_path=$qrel_output_path \ % output dir
      --output_trec_file=$output_trec_file \
      --trec_gold_qrel_file_path=$trec_gold_qrel_file_path \ % gold qrel file
      --per_gpu_train_batch_size=4 \ 
      --test_type=convqa
      --max_query_length=32 \
      --max_doc_length=384 \ 
      --max_response_length=64 \
      --max_concat_length=512 \ 
      --dataset="topiocqa" \
