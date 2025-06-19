import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import io
import matplotlib.pyplot as plt

# ... 其他 import ...

# 設定全域字體大小
plt.rcParams.update({'font.size': 18}) # 將字體大小設置為24 (約是預設字體的2-3倍，具體效果取決於預設值)
# 設定 Matplotlib 顯示中文 (如果運行環境支持)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 例如 'Microsoft JhengHei' 或 'SimHei'
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# 檔案內容片段，模擬從 '熱.txt' 讀取。請注意，這部分內容應該與您實際的檔案內容一致。
file_content = """入口溫度	出口溫度	滾軋速度	帶條厚度	材料分類	抗變形能力	摩擦係數	軋輥直徑	減少率	應變率	潤滑形式	材料編號	潤滑形式編號	力矩
1174.84	941.98	2.72	24.54	A36	154.86	0.245	653.22	0.36	0.58	oil	0	1	1222.8
1143.09	927.74	1.7	28.45	A36	108.36	0.215	640.43	0.3	0.8	oil	0	1	1089.33
1182.38	901.79	2.55	32.06	A36	161.06	0.23	668.79	0.36	0.74	water-based	0	2	1224.46
1226.15	880.59	1.83	22.06	AISI 304	139.04	0.106	647.60	0.40	0.66	dry	1	0	1101.68
1138.29	920.95	3.92	28.71	SS400	154.88	0.169	649.33	0.28	0.88	oil	2	1	1119.4
1101.65	881.20	2.34	24.03	AISI 304	131.21	0.173	621.36	0.22	0.79	water-based	1	2	894.22
1147.61	925.87	3.89	22.98	SS400	172.38	0.228	665.55	0.27	0.92	dry	3	0	1092.48
1149.82	928.59	2.48	23.70	A36	144.11	0.126	634.81	0.28	0.76	dry	0	0	1048.2
1092.08	915.39	2.40	24.60	AISI 304	179.06	0.186	644.61	0.36	0.87	dry	1	0	1232.49
1225.17	921.75	2.20	28.33	AISI 304	180.57	0.155	676.83	0.24	0.89	oil	1	1	1200.75
"""

# 使用 io.StringIO 讀取字符串數據，指定分隔符為 '\t' (tab)
df = pd.read_csv(io.StringIO(file_content), sep='\t')

print("--- 1. 資料準備 ---")
# 定義特徵 (X) 和目標變數 (y)
# 目標變數：材料分類
y = df['材料分類']

# 特徵：排除 '材料分類' (目標), '材料編號' (與目標相關聯的編號，避免資料洩露), '潤滑形式' (使用其編號形式)
X = df.drop(columns=['材料分類', '材料編號', '潤滑形式'])

# 對目標變數進行 Label Encoding，因為分類模型需要數值型標籤
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_ # 獲取類別名稱，用於後續的可視化

print("選定的特徵欄位：")
print(X.columns.tolist())
print(f"目標變數 (y) 原始唯一值：{y.unique()}")
print(f"目標變數 (y) 編碼後對應：{list(zip(class_names, le.transform(class_names)))}")

# 分割資料為訓練集和測試集
# 調整 test_size 以確保測試集大小足夠大，能夠包含所有類別的樣本 (共 3 類)
# 由於數據量極小 (10 筆)，使用 test_size=0.4
# 這將導致訓練集有 6 筆資料，測試集有 4 筆資料。
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded)

print(f"\n訓練集大小：{X_train.shape[0]} 筆資料")
print(f"測試集大小：{X_test.shape[0]} 筆資料")

print("\n--- 2. 建立和訓練隨機森林分類模型 ---")
# 初始化隨機森林分類模型
# n_estimators: 森林中樹的數量 (通常越多越好，但會增加計算成本)
# random_state: 設定種子以確保結果可重現
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 訓練模型
print("開始訓練隨機森林分類模型...")
rf_classifier.fit(X_train, y_train)
print("模型訓練完成。")

print("\n--- 3. 模型預測 ---")
y_pred = rf_classifier.predict(X_test)
print(f"測試集實際標籤 (部分)：{y_test.tolist()}")
print(f"測試集預測標籤 (部分)：{y_pred.tolist()}")
# 將預測的數值標籤轉換回原始的材料分類名稱
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)
print(f"測試集實際材料分類 (部分)：{y_test_labels.tolist()}")
print(f"測試集預測材料分類 (部分)：{y_pred_labels.tolist()}")


print("\n--- 4. 模型評估 ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"準確率 (Accuracy): {accuracy:.2f}")

print("\n分類報告 (Classification Report):")
print(classification_report(y_test, y_pred, target_names=class_names))

# 特徵重要性
print("\n特徵重要性：")
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X.columns)
# 按照重要性降序排序
feature_importances = feature_importances.sort_values(ascending=False)
print(feature_importances)

# 可視化特徵重要性
plt.figure(figsize=(10, 7))
sns.barplot(x=feature_importances.values, y=feature_importances.index, palette='viridis')
plt.title('特徵重要性')
plt.xlabel('重要性分數')
plt.ylabel('特徵')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout() # 自動調整佈局，防止標籤重疊
plt.show()

print("\n隨機森林分類模型程式執行完成。")