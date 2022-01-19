import transformers
from datasets import load_dataset
import nltk
from rouge import Rouge
import nltk
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_rouge(predictions, references):
    rouge = Rouge()
    preprocess_exs = lambda exs : [ex.strip().lower() for ex in exs]
    rouge_scores =  rouge.get_scores(preprocess_exs(predictions), preprocess_exs(references), avg=True)
    return {k: v['f'] for k, v in rouge_scores.items()}


print(transformers.__version__)

nltk.download('punkt')

model_checkpoint = "sberbank-ai/ruT5-base"

print("LOAD DATASET")
raw_datasets = load_dataset('csv', data_files='two_chats_df_train.csv')
print("LOAD TOKENIZER")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 256
max_target_length = 64

prefix = "ответ: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["Context"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["Response"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("TOKENIZE")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
tokenized_datasets_train_test = tokenized_datasets.copy()
test_ds = tokenized_datasets_train_test["train"].train_test_split(test_size=0.1)
tokenized_datasets_train_test['validation'] = test_ds['test']
tokenized_datasets_train_test['train'] = test_ds['train']

print(tokenized_datasets_train_test)

del tokenized_datasets
del test_ds

print("LOAD MODEL")
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 8
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-plenka-chatbot-full",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=50,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets_train_test["train"],
    eval_dataset=tokenized_datasets_train_test["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)
print("TRAIN")
trainer.train()
print("SAVE LOCALLY")
model.save_pretrained('model-plenka-chatbot')
tokenizer.save_pretrained('model-plenka-chatbot')

print("SAVE ON HUB")
model.push_to_hub("valeriazen/ruT5-base-finetuned-plenka-chatbot-full")
tokenizer.push_to_hub("valeriazen/ruT5-base-finetuned-plenka-chatbot-full")
