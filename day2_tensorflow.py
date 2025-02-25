from transformers import BertTokenizer,TFBertModel

#Load a Pre-Trained Model BERT Tokenizer and Model

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = TFBertModel.from_pretrained("bert-base-uncased")

#Tokenize a Sample Input
text = "Transformer are Powerfull Model for NLP task"
inputs= tokenizer(text,return_tensors='tf')

#Pass the Inputs throgh Model
outputs = model(**inputs)
print("HIdden States Shape:",outputs.last_hidden_state.shape)
