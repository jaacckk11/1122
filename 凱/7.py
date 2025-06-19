import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 設定中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 16})

# 產生模擬數據（500 筆樣本）
np.random.seed(42)
n_samples = 500

materials = ['A36', 'AISI304', 'SS400']
lubricants = ['dry', 'oil', 'water-based']
material_map = {m: i for i, m in enumerate(materials)}
lubricant_map = {l: i for i, l in enumerate(lubricants)}

df = pd.DataFrame({
    '入口溫度': np.random.uniform(1050, 1250, n_samples),
    '出口溫度': np.random.uniform(850, 950, n_samples),
    '滾軋速度': np.random.uniform(1.5, 4.0, n_samples),
    '帶條厚度': np.random.uniform(20, 35, n_samples),
    '材料分類': np.random.choice(materials, n_samples),
    '抗變形能力': np.random.uniform(100, 200, n_samples),
    '摩擦係數': np.random.uniform(0.1, 0.25, n_samples),
    '軋輥直徑': np.random.uniform(620, 680, n_samples),
    '減少率': np.random.uniform(0.2, 0.4, n_samples),
    '應變率': np.random.uniform(0.5, 1.0, n_samples),
    '潤滑形式': np.random.choice(lubricants, n_samples)
})

df['材料編號'] = df['材料分類'].map(material_map)
df['潤滑形式編號'] = df['潤滑形式'].map(lubricant_map)
df['力矩'] = (
    df['抗變形能力'] * df['減少率'] * df['摩擦係數'] * df['軋輥直徑'] / 10 +
    np.random.normal(0, 20, n_samples)
)

# 特徵與目標變數
y = df['材料分類']
X = df.drop(columns=['材料分類', '材料編號', '潤滑形式'])

# 標籤編碼
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# 訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("準確率：", accuracy_score(y_test, y_pred))
print("\n分類報告：")
print(classification_report(y_test, y_pred, target_names=class_names))

# 混淆矩陣視覺化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('混淆矩陣 (500 筆樣本)')
plt.xlabel('預測類別')
plt.ylabel('實際類別')
plt.tight_layout()
plt.show()
