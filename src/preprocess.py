import os
import re
import jieba
import json
from collections import Counter

# ------------------------
# 路径配置
# ------------------------
RAW_DIR = "../data/raw/"         # 原始文本目录
CLEANED_DIR = "../data/cleaned/" # 清洗后文本目录
OUTPUT_DIR = "../output/"        # 分词与统计输出目录
STOPWORDS_PATH = "../stopwords/stopwords.txt"

os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# 1. 读取停用词
# ------------------------
def load_stopwords(path):
    with open(path, encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f if line.strip()])
    return stopwords

stopwords = load_stopwords(STOPWORDS_PATH)

# ------------------------
# 2. 文本清洗函数
# ------------------------
def clean_text(text: str) -> str:
    # 去除非中文、英文、数字及常用标点
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9。！？,.，、\s]", "", text)
    # 合并多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------
# 3. 遍历原始文本进行清洗
# ------------------------
for filename in os.listdir(RAW_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned_text = clean_text(raw_text)

        with open(os.path.join(CLEANED_DIR, filename), "w", encoding="utf-8") as f:
            f.write(cleaned_text)

print("文本清洗完成，结果已保存至 data/cleaned/")

# ------------------------
# 4. 分词 + 停用词处理
# ------------------------
for filename in os.listdir(CLEANED_DIR):
    if filename.endswith(".txt"):
        with open(os.path.join(CLEANED_DIR, filename), "r", encoding="utf-8") as f:
            text = f.read()

        # 使用jieba分词并去停用词
        words = [
            w for w in jieba.cut(text)
            if w not in stopwords and len(w.strip()) > 1
        ]

        # 保存分词结果
        seg_file = os.path.join(OUTPUT_DIR, filename.replace(".txt","_seg.txt"))
        with open(seg_file, "w", encoding="utf-8") as f:
            f.write(" ".join(words))

print("分词+停用词处理完成，结果已保存至 output/")

# ------------------------
# 5. 高频词统计（可选）
# ------------------------
word_freq_summary = {}

for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith("_seg.txt"):
        with open(os.path.join(OUTPUT_DIR, filename), "r", encoding="utf-8") as f:
            words = f.read().split()

        counter = Counter(words)
        word_freq_summary[filename] = counter.most_common(50)  # 统计前50高频词

# 保存高频词统计结果
with open(os.path.join(OUTPUT_DIR, "word_freq.json"), "w", encoding="utf-8") as f:
    json.dump(word_freq_summary, f, ensure_ascii=False, indent=2)

print("高频词统计完成，结果已保存至 output/word_freq.json")
