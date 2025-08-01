import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import platform

# 路径
SEG_DIR = "../output/"
VIS_DIR = "../output/visual/"
os.makedirs(VIS_DIR, exist_ok=True)

# 自动选择中文字体
if platform.system() == "Windows":
    plt.rcParams['font.sans-serif'] = ['SimHei']
elif platform.system() == "Darwin":
    plt.rcParams['font.sans-serif'] = ['STHeiti']
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'AR PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载分词语料
corpus = []
file_names = []
for filename in os.listdir(SEG_DIR):
    if filename.endswith("_seg.txt"):
        file_names.append(filename)
        with open(os.path.join(SEG_DIR, filename), "r", encoding="utf-8") as f:
            corpus.append(f.read())

doc_count = len(corpus)
print(f"共加载文档数: {doc_count}")
if doc_count == 0:
    raise ValueError("未找到分词文本，请先运行 preprocess.py 生成 _seg.txt 文件。")

# 2. TF-IDF 特征提取
vectorizer = TfidfVectorizer(max_features=5000)  # 增大特征数
tfidf_matrix = vectorizer.fit_transform(corpus)
print(f"TF-IDF矩阵形状: {tfidf_matrix.shape}")

# 3. LDA主题建模
n_topics = min(3, doc_count)  # 主题数最多不超过文档数
lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_topics = lda_model.fit_transform(tfidf_matrix)

# 4. 输出主题关键词
def print_top_words(model, feature_names, n_top_words=10):
    topics_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_features]
        topics_keywords.append(top_words)
        print(f"主题 {topic_idx+1}: {' '.join(top_words)}")
    return topics_keywords

feature_names = vectorizer.get_feature_names_out()
topics_keywords = print_top_words(lda_model, feature_names)

# 保存主题关键词表
pd.DataFrame(topics_keywords).to_csv(
    os.path.join(VIS_DIR, "topic_keywords.csv"),
    index_label="主题编号",
    encoding="utf-8-sig"
)

# 5. 保存文档主题分布
doc_topic_df = pd.DataFrame(lda_topics, columns=[f"主题{i+1}" for i in range(n_topics)])
doc_topic_df.insert(0, "文档名", file_names)
doc_topic_df.to_csv(
    os.path.join(VIS_DIR, "document_topic_distribution.csv"),
    index=False,
    encoding="utf-8-sig"
)
print(f"文档主题分布已保存: {VIS_DIR}document_topic_distribution.csv")

# 6. 文档聚类（仅当文档数≥2）
if doc_count >= 2:
    kmeans = KMeans(n_clusters=min(3, doc_count), random_state=42)
    clusters = kmeans.fit_predict(lda_topics)

    plt.figure(figsize=(8, 6))
    plt.scatter(lda_topics[:, 0], lda_topics[:, 1], c=clusters, cmap='rainbow')
    for i, txt in enumerate(file_names):
        plt.annotate(txt.replace("_seg.txt", ""), (lda_topics[i, 0], lda_topics[i, 1]))
    plt.title("文档主题分布聚类可视化")
    plt.xlabel("主题1权重")
    plt.ylabel("主题2权重")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "topic_clusters.png"), dpi=300)
    plt.show()
    print(f"文档聚类结果已保存: {VIS_DIR}topic_clusters.png")
