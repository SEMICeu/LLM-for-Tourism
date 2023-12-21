from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import BertTokenizerFast, RobertaTokenizerFast
from transformers import BertForMaskedLM, RobertaForMaskedLM
from sagemaker.s3 import S3Downloader
from sagemaker.session import Session
from transformers import BertConfig, RobertaConfig
from sagemaker.s3 import S3Uploader
from sagemaker.s3 import parse_s3_url
from transformers import pipeline 
from datasets import load_dataset
from datasets import DatasetDict
from transformers import Trainer
from pathlib import Path
import pandas as pd
import transformers
import sagemaker
import datasets
import argparse
import logging
import random
import pandas as pd
import shutil
import torch
import boto3
import json
import time
import math
import sys
import os


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.getLevelName('INFO'), 
                    handlers=[logging.StreamHandler(sys.stdout)], 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Log versions of dependencies
logger.info(f'[Using Transformers: {transformers.__version__}]')
logger.info(f'[Using SageMaker: {sagemaker.__version__}]')
logger.info(f'[Using Datasets: {datasets.__version__}]')
logger.info(f'[Using Torch: {torch.__version__}]')
logger.info(f'[Using Boto3: {boto3.__version__}]')
logger.info(f'[Using Pandas: {pd.__version__}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    logger.info('Parsing command line arguments')
    parser.add_argument('--input_dir', type=str, default=os.environ['SM_INPUT_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    #parser.add_argument('--master_host', type=str, default=os.environ['SMDATAPARALLEL_SERVER_ADDR'])
    
    # [IMPORTANT] Hyperparameters sent by the client (Studio notebook with the driver code to launch training) 
    # are passed as command-line arguments to the training script
    parser.add_argument('--s3_bucket', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--num_train_epochs', type=int)
    parser.add_argument('--per_device_train_batch_size', type=int)
    parser.add_argument('--region', type=str)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--corpus', type=str)
    parser.add_argument('--datasets', type=str)
    
    args, _ = parser.parse_known_args()
    current_host = args.current_host
    #master_host = args.master_host
    
    logger.info(f'Current host = {current_host}')
    #logger.info(f'Master host = {master_host}')
    
    S3_BUCKET = args.s3_bucket
    MAX_LENGTH = args.max_len
    CHUNK_SIZE = args.chunk_size
    TRAIN_EPOCHS = args.num_train_epochs
    BATCH_SIZE = args.per_device_train_batch_size
    REGION = args.region 
    SAVE_STEPS = 10000
    SAVE_TOTAL_LIMIT = 2
    vocab = args.vocab
    corpus = args.corpus
    datasets = args.datasets
    
    LOCAL_DATA_DIR = '/tmp/cache/data/processed'
    LOCAL_MODEL_DIR = '/tmp/cache/model/custom'
    
    config = RobertaConfig()
    
    # Setup SageMaker Session for S3Downloader and S3Uploader 
    boto_session = boto3.session.Session(region_name=REGION)
    sm_session = sagemaker.Session(boto_session=boto_session)
    
        
    def upload(ebs_path: str, s3_path: str, session: Session) -> None:
        S3Uploader.upload(ebs_path, s3_path, sagemaker_session=session)
        
    
    # Download saved custom vocabulary file from S3 to local input path of the training cluster
    logger.info(f'Downloading custom vocabulary from [{S3_BUCKET}/data/vocab/] to [{args.input_dir}/vocab/]')
    path = os.path.join(f'{args.input_dir}', 'vocab1')
    os.mkdir(path)
    path2 = os.path.join(f'{args.input_dir}', 'corpus')
    os.mkdir(path2)

    vocab = pd.read_json(f's3://{S3_BUCKET}/data/vocab/vocab.json', orient = 'index')
    vocab = vocab.to_dict()
    with open(f'{path}/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab[0], f)
        
    vocab = pd.read_csv(f's3://{S3_BUCKET}/data/vocab/merges.csv')
    with open(f'{path}/merges.txt', 'w', encoding='utf-8') as f:
        for content in vocab['0'].values:
            f.write(str(content) + '\n')
            
    corpus = pd.read_csv(f's3://{S3_BUCKET}/data/corpus/corpus.csv')
    with open(f'{path2}/corpus.txt', 'w', encoding='utf-8') as f:
        for ID, content in zip(corpus['Unnamed: 0'].values, corpus['0'].values):
            f.write('\n'.join([str(ID), content]))
        
         
    # Download preprocessed datasets from S3 to local EBS volume (cache dir)
    logger.info(f'Downloading preprocessed datasets from [{S3_BUCKET}/data/processed/] to [{LOCAL_DATA_DIR}/]')
    
    
    # Re-create BERT WordPiece tokenizer 
    logger.info(f'Re-creating ROBERTA tokenizer using custom vocabulary from [{args.input_dir}/vocab1/]')
    tokenizer = RobertaTokenizerFast.from_pretrained(path, config=config)
    tokenizer.model_max_length = MAX_LENGTH
    tokenizer.init_kwargs['model_max_length'] = MAX_LENGTH
    logger.info(f'Tokenizer: {tokenizer}')

    
    # Read dataset and collate to create mini batches for Masked Language Model (MLM) training
    logger.info('Reading and collating input data to create mini batches for Masked Language Model (MLM) training')
    dataset = load_dataset('text', data_files=f'{path2}/corpus.txt', split='train', cache_dir='/tmp/cache')
    logger.info(f'Dataset: {dataset}')

    # Split dataset into train and validation splits 
    logger.info('Splitting dataset into train and validation splits')
    train_test_splits = dataset.train_test_split(shuffle=True, seed=123, test_size=0.1)
    data_splits = DatasetDict({'train': train_test_splits['train'], 
                               'validation': train_test_splits['test']})
    logger.info(f'Data splits: {data_splits}')
    
    
    # Tokenize dataset
    def tokenize(article, tokenizer = tokenizer):
        tokenized_article = tokenizer(article['text'])
        if tokenizer.is_fast:
            tokenized_article['word_ids'] = [tokenized_article.word_ids(i) for i in range(len(tokenized_article['input_ids']))]
        return tokenized_article


    logger.info('Tokenizing dataset splits')
    N_GPUS = 1
    num_proc = int(os.cpu_count()/N_GPUS)
    logger.info(f'Total number of processes = {num_proc}')
    tokenized_datasets = data_splits.map(tokenize, batched=True, num_proc=num_proc, remove_columns=['text'])
    logger.info(f'Tokenized datasets: {tokenized_datasets}')
    
    # Concat and chunk dataset 
    def concat_and_chunk(articles, CHUNK_SIZE = CHUNK_SIZE):
        # Concatenate all texts
        concatenated_examples = {key: sum(articles[key], []) for key in articles.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(articles.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length//CHUNK_SIZE) * CHUNK_SIZE
        # Split by chunks of max_len
        chunked_articles = {key: [text[i : i+CHUNK_SIZE] for i in range(0, total_length, CHUNK_SIZE)] for key, text in concatenated_examples.items()}
        # Create a new labels column
        chunked_articles['labels'] = chunked_articles['input_ids'].copy()
        return chunked_articles

    logger.info('Concatenating and chunking the datasets to a fixed length')
    chunked_datasets = tokenized_datasets.map(concat_and_chunk, batched=True, num_proc=num_proc)
    logger.info(f'Chunked datasets: {chunked_datasets}')
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                    mlm=True, 
                                                    mlm_probability=0.15)
    
    # Load MLM
    logger.info('Loading RobertaForMaskedLM model')
    mlm = RobertaForMaskedLM(config=config)
    
    # Train MLM
    logger.info('Training MLM')
    training_args = TrainingArguments(output_dir='/tmp/checkpoints', 
                                      overwrite_output_dir=True, 
                                      optim='adamw_torch',
                                      num_train_epochs=TRAIN_EPOCHS,
                                      per_device_train_batch_size=BATCH_SIZE,
                                      evaluation_strategy='epoch',
                                      save_steps=SAVE_STEPS, 
                                      save_total_limit=SAVE_TOTAL_LIMIT)
    trainer = Trainer(model=mlm, 
                      args=training_args, 
                      data_collator=data_collator,
                      train_dataset=chunked_datasets['train'],
                      eval_dataset=chunked_datasets['validation'])
    
    # Evaluate trained model for perplexity
    eval_results = trainer.evaluate()
    logger.info(f"Perplexity before training: {math.exp(eval_results['eval_loss']):.2f}")
    
    trainer.train()
    
    eval_results = trainer.evaluate()
    logger.info(f"Perplexity after training: {math.exp(eval_results['eval_loss']):.2f}")
    
    
    #if current_host == master_host:
        
    if not os.path.exists(LOCAL_MODEL_DIR):
        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

    # Save trained model to local model directory
    logger.info(f'Saving trained MLM to [/tmp/cache/model/custom/]')
    trainer.save_model(LOCAL_MODEL_DIR)

    if os.path.exists(f'{LOCAL_MODEL_DIR}/pytorch_model.bin') and os.path.exists(f'{LOCAL_MODEL_DIR}/config.json'):
        # Copy trained model from local directory of the training cluster to S3 
        logger.info(f'Copying saved model from local to [s3://{S3_BUCKET}/model/custom/]')
        upload(f'{LOCAL_MODEL_DIR}/', f's3://{S3_BUCKET}/model2/custom/', sm_session)

        # Copy vocab.txt to local model directory - this is needed to re-create the trained MLM
        logger.info('Copying custom vocabulary to local model artifacts location to faciliate model evaluation')
        shutil.copyfile(f'{path}/vocab.txt', f'{LOCAL_MODEL_DIR}/vocab.txt')

        # Copy vocab.txt to saved model artifacts location in S3
        logger.info(f'Copying custom vocabulary from [{path}/vocab.txt] to [s3://{S3_BUCKET}/model/custom/] for future stages of ML pipeline')
        upload(f'{path}/', f's3://{S3_BUCKET}/model2/custom/', sm_session)