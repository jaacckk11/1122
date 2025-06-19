import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 混淆矩陣數據
labels = ['A36', 'AISI 304', 'AISI 316', 'SS400']
conf_matrix = np.array([
    [250,   0,  31,   0],  # Actual A36
    [  0,  63,   0,   0],  # Actual AISI 304
    [  0,   0, 125,   0],  # Actual AISI 316
    [  0,   0,   0,  94]   # Actual SS400
])

# 繪製混淆矩陣
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("預測標籤")
plt.ylabel("實際標籤")
plt.title("材料分類混淆矩陣（樣本數 = 500）")
plt.tight_layout()
plt.show()
