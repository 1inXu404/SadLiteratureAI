import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# =========================
# 路径配置
# =========================
SEG_DIR = "../output/"         # 分词后文件目录
VIS_DIR = "../output/visual/"  # 可视化结果目录
DICT_DIR = "../sentiment_dict/"# NTUSD词典目录
os.makedirs(VIS_DIR, exist_ok=True)

# =========================
# 1. 加载NTUSD情感词典
# =========================
def load_ntusd_dict():
    pos_file = os.path.join(DICT_DIR, "NTUSD_positive_simplified.txt")
    neg_file = os.path.join(DICT_DIR, "NTUSD_negative_simplified.txt")

    with open(pos_file, encoding="utf-8") as f:
        positive_words = set([line.strip() for line in f if line.strip()])
    with open(neg_file, encoding="utf-8") as f:
        negative_words = set([line.strip() for line in f if line.strip()])

    print(f"正向词数: {len(positive_words)}, 负向词数: {len(negative_words)}")
    return positive_words, negative_words

positive_words, negative_words = load_ntusd_dict()

# =========================
# 2. 加载分词结果
# =========================
corpus_words = []
for filename in os.listdir(SEG_DIR):
    if filename.endswith("_seg.txt"):
        with open(os.path.join(SEG_DIR, filename), "r", encoding="utf-8") as f:
            corpus_words.extend(f.read().split())

print(f"共加载词语数量：{len(corpus_words)}")

# =========================
# 3. 情绪统计与分段分析
# =========================
segment_size = 100  # 每100词为一段
emotion_scores = []
pos_count = 0
neg_count = 0

for i in range(0, len(corpus_words), segment_size):
    segment = corpus_words[i:i+segment_size]
    pos = sum(1 for w in segment if w in positive_words)
    neg = sum(1 for w in segment if w in negative_words)

    emotion_scores.append(pos - neg)
    pos_count += pos
    neg_count += neg

# =========================
# 4. 绘制情绪曲线
# =========================
plt.figure(figsize=(12,5))
plt.plot(range(len(emotion_scores)), emotion_scores, marker='o', color="purple")
plt.axhline(0, color="gray", linestyle="--")
plt.title("基于NTUSD的文本情绪曲线（正数偏积极，负数偏消极）")
plt.xlabel("文本段编号")
plt.ylabel("情绪得分")
plt.grid(alpha=0.3)
plt.savefig(os.path.join(VIS_DIR, "ntusd_emotion_curve.png"), dpi=300)
plt.show()

print("情绪曲线已保存：output/visual/ntusd_emotion_curve.png")

# =========================
# 5. 输出情绪占比饼图
# =========================
plt.figure(figsize=(5,5))
plt.pie(
    [pos_count, neg_count],
    labels=["正向情绪词", "负向情绪词"],
    autopct='%1.1f%%',
    colors=["#66c2a5", "#fc8d62"]
)
plt.title("NTUSD情绪词占比")
plt.savefig(os.path.join(VIS_DIR, "ntusd_emotion_pie.png"), dpi=300)
plt.show()

print("情绪占比饼图已保存：output/visual/ntusd_emotion_pie.png")
