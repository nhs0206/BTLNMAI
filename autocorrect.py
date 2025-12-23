import tkinter as tk
from tkinter import messagebox, scrolledtext
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

root = tk.Tk()
# 1. Cấu hình mô hình
MODEL_NAME = "bmd1905/vietnamese-correction-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

# 2. Hàm xử lý sửa lỗi
def correct_spelling():
    input_text = text_input.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Thông báo", "Vui lòng nhập văn bản cần sửa!")
        return
    
    try:
        # Hiển thị trạng thái đang xử lý
        status.config(text="Trạng thái: Đang xử lý...", fg="orange")
        root.update_idletasks()

        # Tokenize và Generate
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=256, num_beams=5)
        
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Hiển thị kết quả
        text_output.config(state=tk.NORMAL) # Mở khóa để ghi
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, corrected_text)
        text_output.config(state=tk.DISABLED) # Khóa lại sau khi ghi
        status.config(text = "Trạng thái: Thành công!", fg="green")
        
    except Exception as e:
        messagebox.showerror("Lỗi", f"Có lỗi xảy ra: {str(e)}")
        status.config(text="Trạng thái: Lỗi!", fg="red")

# 3. Thiết lập giao diện Tkinter
# root = tk.Tk()
root.title("Autocorrect")
root.geometry("600x550")
root.configure(bg="#272727")

# Tiêu đề
lbl_title = tk.Label(root, text="Sửa lỗi chính tả", font=("Arial", 16, "bold"), bg="#272727", fg="#ffffff")
lbl_title.pack(pady=15)

# Khung Input
tk.Label(root, text="Nhập văn bản lỗi:", font=("Arial", 10, "bold"), bg="#272727", fg="#ffffff").pack(anchor="w", padx=20)
text_input = scrolledtext.ScrolledText(root, height=8, width=65, font=("Arial", 11))
text_input.pack(pady=10, padx=20)

# Nút bấm
btn_correct = tk.Button(root, text="Correct", command= correct_spelling, bg="red", fg="white", font=("Arial", 11, "bold"), padx=10, pady=5, cursor="hand2")
btn_correct.pack(pady=15)

# Khung Ouput
tk.Label(root, text="Kết quả đã sửa:", font=("Arial", 10, "bold"), bg="#272727", fg="#ffffff").pack(anchor="w", padx=20)
text_output = scrolledtext.ScrolledText(root, height=8, width=65, font=("Arial", 11))
text_output.config(state=tk.DISABLED) # Mặc định khóa không cho nhập vào ô kết quả
text_output.pack(pady=5, padx=20)

status = tk.Label(root, text="Trạng thái: Sẵn sàng", font=("Arial", 9, "italic"), bg="#272727", fg="#ffffff")
status.pack(pady=10)


root.mainloop()