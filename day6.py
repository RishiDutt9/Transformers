from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset


#Load Dataset

dataset = load_dataset("ag_news")


#Load RoBERTa tokenizer and Model

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=4)


#Tokenize the dataset

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",max_length =128, truncation=True)


tokenize_datasets = dataset.map(tokenize_function, batched=True)

#Pepare the dataset for training

tokenize_datasets = tokenize_datasets.remove_columns(["text"])
tokenize_datasets = tokenize_datasets.rename_column("label", "labels")
tokenize_datasets.set_format("torch")


train_dataset = tokenize_datasets["train"]
test_dataset = tokenize_datasets["test"]

#Training Arguments

training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 3,
    weight_decay = 0.01,
    
    save_steps = 500
)


#Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    processing_class = tokenizer
)

#Train the model

trainer.train()

#Evaluate the model

results  = trainer.evaluate()

print("Evaluation results : ", results)


import deepsee.ai as ds