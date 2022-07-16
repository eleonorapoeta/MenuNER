import argparse
import json
import spacy
import torch
import random
from transformers import BertForPreTraining
from transformers import AdamW
from transformers import Trainer, TrainingArguments
from bert_training.bert_domain_utils import input_data_prep, FoodDomainDataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--filtered_reviews_filename', type=str, default='../data/filtered_reviews.json')
    parser.add_argument('--checkpoint_path', type=str, default='../bert_checkpoint/')
    params = parser.parse_args()

    with open(params.reviews_filename, 'r') as f:
        reviews = json.load(f)

    model = BertForPreTraining.from_pretrained("bert-base-cased")

    # Sentence preparation for NSP
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')

    set_of_sent = []
    for rev in nlp.pipe(reviews, disable=['parser', 'tagger', 'ner'], batch_size=1000, n_process=2):
        x = [sent.text.strip() for sent in rev.sents]
        set_of_sent.append(x)

    sentences_a = []
    sentences_b = []
    labels = []

    for rev in set_of_sent:
        num_sent = len(rev)
        if len(rev) > 1:
            start = random.randint(0, num_sent - 2)
            sentences_a.append(rev[start])
            if random.random() > 0.5:
                sentences_b.append(rev[start + 1])
                labels.append(0)
            else:
                sentences_b.append(set_of_sent[random.randint(0, len(set_of_sent) - 1)])
                labels.append(1)

    # MLM preparation
    eval_samples = int(len(labels) * 0.9)

    train_inputs = input_data_prep(sentences_a[:eval_samples], sentences_b[:eval_samples], labels[:eval_samples])
    eval_inputs = input_data_prep(sentences_a[eval_samples:], sentences_b[eval_samples:], labels[eval_samples:])

    # Dataset creation
    train_dataset = FoodDomainDataset(train_inputs)
    eval_dataset = FoodDomainDataset(eval_inputs)

    # Model preparation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    optim = AdamW(model.parameters(), eps=1e-8, lr=3e-5)
    training_args = TrainingArguments(
        output_dir='../bert_checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        learning_rate=3e-5,
        warmup_steps=10,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        evaluation_strategy="steps",
        save_strategy="steps",
        fp16=True,
        eval_steps=10000,
        save_steps=10000
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        optimizers=(optim, None))

    if params.checkpoint_path is not None:
        trainer.train(params.checkpoint_path)
    else:
        trainer.train()

    trainer.model.save_pretrained('../bert_checkpoint')
