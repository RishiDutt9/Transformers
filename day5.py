from transformers import AutoTokenizer ,TrainingArguments, Trainer,AutoModelForSequenceClassification
from datasets import load_dataset


#Load THe Datasets

dataset = load_dataset("imdb")

#Tokenizer

Tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#Tokenize the Dataset
def toeknize_function(examples):
    return Tokenizer(examples["text"],padding = "max_length",truncation = True)

tokenize_datasets = dataset.map(toeknize_function,batched = True)

#Prepare the Model

tokenize_datasets = tokenize_datasets.remove_columns(["text"])
tokenize_datasets = tokenize_datasets.rename_column("label","labels")
tokenize_datasets.set_format("torch")

train_dataset = tokenize_datasets["train"]

train_dataset = tokenize_datasets["train"]
test_dataset = tokenize_datasets["test"]

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels = 2)

training_args = TrainingArguments(
    output_dir = "./results",
    eval_strategy = "epoch",
    learning_rate = 2e-5,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 8,
    num_train_epochs = 3,
    weight_decay = 0.01,
    logging_dir = "./logs",
    logging_steps = 10,
    save_steps = 500

)


trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = test_dataset,
    processing_class = Tokenizer
)

trainer.train()


#Evaluate the model

results = trainer.evaluate()
print("Evaluation results : ",results)


#Experiment with GPT

from transformers import AutoModelForCausalLM

gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")

input_text = "Once upon a time a cat"
input_ids = Tokenizer.encode(input_text,return_tensors = "pt")
output = gpt_model.generate(input_ids, max_length = 50,num_return_sequences = 1)


print("GEnerated Text",Tokenizer.decode(output[0],skip_special_tokens = True))