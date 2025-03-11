from collect_data import ds

import evaluate
import numpy as np

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

def tokenize_and_align_labels(examples):
    sentences, sentence_tags = [], []
    labels = []

    for tokens, pols in zip(examples["Tokens"], examples["Polarities"]):
        bert_tokens = []
        bert_att = []
        pols_label = 0

        for i in range( len(tokens) ):
            t = tokenizer.tokenize( tokens[i] )

            bert_tokens += t
            if int( pols[i] ) != -1:
                bert_att += t
                pols_label = int( pols[i] )
        
        sentences.append(" ".join( bert_tokens ))
        sentence_tags.append(" ".join( bert_att ))
        labels.append( pols_label )

    tokenized_inputs = tokenizer(sentences, sentence_tags, padding=True, truncation=True, return_tensors="pt")
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    preprocessed_ds = ds.map(tokenize_and_align_labels, batched=True)

    accuracy = evaluate.load("accuracy")

    id2label = {
        0: 'Negative', 
        1: 'Neutral',
        2: 'Positive'
    }
    label2id = {
        'Negative': 0, 
        'Neutral': 1, 
        'Positive': 2
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir= "./bert_ATSA_1",
        learning_rate= 2e-5,
        per_device_train_batch_size= 128,
        per_device_eval_batch_size= 128,
        num_train_epochs= 50,
        weight_decay= 0.01,
        eval_strategy= "epoch",
        save_strategy= "epoch",
        logging_strategy= "epoch",
        load_best_model_at_end= True,
        metric_for_best_model= "accuracy",
        # report_to="wandb",
    )

    trainer = Trainer(
        model= model,
        args= training_args,
        train_dataset= preprocessed_ds["train"],
        eval_dataset= preprocessed_ds["test"],
        processing_class= tokenizer,
        compute_metrics= compute_metrics,
    )

    trainer.train()
    