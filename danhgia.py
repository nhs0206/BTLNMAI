import torch
import time
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

MODEL_NAME = "bmd1905/vietnamese-correction-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1.Tải mô hình và dữ liệu
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
metric = evaluate.load("sacrebleu")

# Tải tập test (lấy 100 câu ngẫu nhiên để đánh giá)
dataset = load_dataset("bmd1905/vi-error-correction-2.0", split="test")
test_data = dataset.shuffle(seed= 100).select(range(1000))

# 2.Chạy dự đoán
predictions = []
references = []

print("Đang tiến hành đánh giá trên tập dữ liệu mẫu...")
start_time = time.time()
for example in tqdm(test_data):
    input_text = example["input"]
    target_text = example["output"]
    
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
    
    # Sinh văn bản (Inference)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5)
    
    # Giải mã kết quả
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    predictions.append(pred_text)
    references.append([target_text]) # Sacrebleu yêu cầu list của list cho references
end_time = time.time()


total_time = end_time - start_time
avg_time = total_time / len(test_data)
fps = 1 / avg_time

# 3.Tính toán kết quả cuối cùng
bleu_score = metric.compute(predictions=predictions, references=references)

print(f"  - Số lượng mẫu thử nghiệm: {len(test_data):<16} ")
print(f"  - Điểm SacreBLEU:         {bleu_score['score']:>8.2f} / 100 ")
print(f"  - Tổng thời gian xử lý:   {total_time:>8.2f} giây   ")
print(f"  - Thời gian TB mỗi câu:   {avg_time:>8.4f} giây   ")
print(f" - Tốc độ xử lý:           {fps:>8.2f} câu/giây ")
print(f" - Thiết bị sử dụng:       {device.upper():>14} ")
    


print("\nVí dụ thực tế:")
for i in range(3):
    print(f"Input: {test_data[i]['input']}")
    print(f"Model: {predictions[i]}")
    print(f"True : {test_data[i]['output']}")
    print("-" * 20)