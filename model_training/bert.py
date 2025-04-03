import datasets
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

from dataloading import get_hasib18_fns

if __name__ == "__main__":
    pass

train_df, test_df = get_hasib18_fns()

train_dataset = datasets.Dataset.from_pandas(train_df)
test_dataset = datasets.Dataset.from_pandas(test_df)

print("loaded dataset")

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
# freeze BERT encoder weights
for param in model.bert.parameters():
    param.requires_grad = False

print("loaded model")

def tokenize(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_test = test_dataset.map(tokenize, batched=True)


training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
)

print("training model")
trainer.train()
print("trained model, evaluating model")
predictions = trainer.predict(tokenized_test)
preds = predictions.predictions.argmax(-1)
labels = predictions.label_ids
f1 = f1_score(labels, preds)  # or 'macro' or 'binary'
print(f"F1 Score: {f1:.4f}")

import matplotlib.pyplot as plt

c_m = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(c_m)
disp.plot()
plt.show()