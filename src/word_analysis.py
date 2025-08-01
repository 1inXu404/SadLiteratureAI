import os
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 路径配置（基于你项目结构）
SEG_DIR = "../output/"         # 分词结果文件在 output/ 目录下（如 001__seg.txt）
VIS_DIR = "../output/visual/"  # 可视化输出目录
os.makedirs(VIS_DIR, exist_ok=True)

# 读取分词结果
all_words = []
for filename in os.listdir(SEG_DIR):
    if filename.endswith("_seg.txt"):
        filepath = os.path.join(SEG_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            words = f.read().split()
            all_words.extend(words)

print(f"共加载词语数量: {len(all_words)}")

# 高频词统计
word_freq = Counter(all_words)
top_words = word_freq.most_common(30)
print("高频词前10:", top_words[:10])

# 设置中文字体，防止图表中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows常用中文字体
plt.rcParams['axes.unicode_minus'] = False

# 1. 高频词条形图
plt.figure(figsize=(12, 6))
words, counts = zip(*top_words)
plt.bar(words, counts, color='skyblue')
plt.xticks(rotation=45)
plt.title("高频词统计前30")
plt.ylabel("词频")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "top_words_bar.png"), dpi=300)
plt.show()
print(f"高频词条形图已保存：{VIS_DIR}top_words_bar.png")

# 2. 词云绘制
# Windows环境字体路径示例，你可以根据需要调整
font_path = "C:/Windows/Fonts/simhei.ttf"

wordcloud = WordCloud(
    font_path=font_path,
    background_color="white",
    width=1200,
    height=800
).generate_from_frequencies(word_freq)

wordcloud.to_file(os.path.join(VIS_DIR, "wordcloud.png"))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
print(f"词云已保存：{VIS_DIR}wordcloud.png")
