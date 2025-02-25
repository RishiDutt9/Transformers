from transformers import BertTokenizer,BertModel

#Load a Pre-Trained Model BERT Tokenizer and Model

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased")

#Tokenize a Sample Input
text = "Transformer are Powerfull Model for NLP task"
inputs= tokenizer(text,return_tensors="pt")

#Pass the Inputs throgh Model
outputs = model(**inputs)
print("HIdden States Shape:",outputs.last_hidden_state.shape)

