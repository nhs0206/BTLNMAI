# %%
# from huggingface_hub import interpreter_login
# interpreter_login()

# %%
import warnings

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

warnings.filterwarnings("ignore")

# %% [markdown]
# # Setup config

# %%
MODEL_NAME = "vinai/bartpho-syllable"
MAX_LENGTH = 256

# %% [markdown]
# # Download dataset

# %%
# Tải dataset
dataset = load_dataset("bmd1905/vi-error-correction-2.0")

# %%
dataset

# %%
# Thu hẹp dữ liệu
dataset["train"] = dataset["train"].select(range(1000))
dataset["test"] = dataset["test"].select(range(1000))

# %%
dataset["train"][-1]

# %%
# Example
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

input = dataset["train"][-1]["input"]
output = dataset["train"][-1]["output"]

# Sử dụng tokenizer để mã hóa dữ liệu đầu vào
inputs = tokenizer(input, text_target=output, max_length=MAX_LENGTH, truncation=True)

inputs, len(inputs["input_ids"])

# %% [markdown]
# # Tokenize dataset

# %%
def preprocess_function(examples):
    # Tokenize dữ liệu đầu vào và apply truncation
    return tokenizer(
        examples["input"],
        text_target=examples["output"],
        max_length=MAX_LENGTH,
        truncation=True,
    )


# Apply tokenization in a batched manner for efficiency
tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# %% [markdown]
# # Model

# %%

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# %% [markdown]
# # Metric

# %%

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Tính toán điểm BLEU (Bilingual Evaluation Understudy Score)

metric = evaluate.load("sacrebleu")

# %%
predictions = [
    "Nếu làm được như vậy thì chắc chắn sẽ không còn trường nào tùy tiện thu tiền cao, gây sự lo lắng của phụ huynh và ai không có tiền thì không cần đóng."
]
references = [
    [
        "Nếu làm được như vậy thì chắc chắn sẽ không còn trường nào tùy tiện thu tiền cao, gây sự lo lắng của phụ huynh và ai không có tiền thì không cần đóng."
    ]
]
metric.compute(predictions=predictions, references=references)

# %%
predictions = [
    "Nếu làm được như vậy thì chắc chắn sẽ khôTng còn trường nà tùy tiện tu tiềncaogây sự lo hắng của phụ huynh và ai khÔng có tiền thì kông cần dong"
]
references = [
    [
        "Nếu làm được như vậy thì chắc chắn sẽ không còn trường nào tùy tiện thu tiền cao, gây sự lo lắng của phụ huynh và ai không có tiền thì không cần đóng."
    ]
]
metric.compute(predictions=predictions, references=references)

# %%
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"sacrebleu": result["score"]}

# %% [markdown]
# # Fine-tuning the model

# %%

args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    num_train_epochs=10, #Số lần đọc dữ liệu của model
    learning_rate=1e-5, #Tốc độ học của model
    per_device_train_batch_size=4, #Quản lý số lượng câu được đưa vào mô hình trong một lần chạy
    per_device_eval_batch_size=4 * 12, #Cập nhật trọng số sau mỗi 4 lần chạy
    gradient_accumulation_steps=4,
    eval_strategy="steps", 
    eval_steps=20000,
    save_strategy="steps",
    logging_steps=20000,
    save_total_limit=5, 
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)

# %%

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics, #Hàm tính điểm BLEU
)

trainer.evaluate()

# %%
trainer.train()

# %%
trainer.evaluate()

# %%
# trainer.push_to_hub(tags="text2text-generation", commit_message="Training complete")