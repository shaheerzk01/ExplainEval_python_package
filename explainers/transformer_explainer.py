import torch
import matplotlib.pyplot as plt


class TransformerExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def explain(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs, output_attentions=True)
        attentions = outputs.attentions
        print("Attention maps:", attentions)
