import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # 假設需要重新分割資料來定義X, y

# --- 假設您已載入資料並準備好 X, y, model_pipeline ---
# 如果您是從頭開始運行，請取消註解以下程式碼區塊
# 這裡使用一個簡化的範例來讓程式碼可執行，實際應用中應使用您的原始數據
# 因為我無法直接訪問您本地的 '熱.txt' 檔案，所以需要一個可執行的範例數據框
# 為了產生類似的樹結構，假設我們有與您圖片中相似的特徵和一個分類目標

# 假設的數據框，與您之前提供的結構類似，但僅包含部分數據以供示範
# 實際應用中請使用您的 '熱.txt' 檔案完整載入
# column_names 定義與您之前提供的一致
column_names = [
    "入口溫度", "出口溫度", "滾軋速度", "帶條厚度", "材料分類", "抗變形能力",
    "摩擦係數", "軋輥直徑", "減少率", "應變率", "潤滑形式", "材料編號",
    "潤滑形式編號", "力矩"
]
# 這裡僅用一個小的範例數據，確保程式碼能運行以產生圖形
# 實際應用中，您會載入完整的 '熱.txt'
data = {
    "入口溫度": [1174.84, 1143.09, 1182.38, 1226.15, 1138.29, 1138.29, 1228.96, 1188.37, 1126.53, 1177.13, 1126.83, 1126.71, 1162.1, 1054.34, 1063.75, 1121.89, 1099.36, 1165.71, 1104.6, 1079.38],
    "出口溫度": [941.98, 927.74, 901.79, 880.59, 920.95, 911.8, 926.86, 919.06, 931.49, 883.94, 939.52, 905.93, 962.26, 879.32, 952.08, 905.94, 880.46, 885.48, 890.39, 912.72],
    "滾軋速度": [2.72, 1.7, 2.55, 1.83, 3.92, 4.34, 1.72, 4.37, 3.07, 2.4, 1.73, 3.0, 3.88, 3.62, 1.65, 1.72, 2.71, 2.39, 2.2, 2.34],
    "帶條厚度": [24.54, 28.45, 32.06, 22.06, 28.71, 27.58, 22.16, 29.36, 24.1, 27.32, 21.23, 26.89, 24.6, 32.34, 20.85, 26.28, 26.88, 30.88, 28.61, 30.01],
    "材料分類": ['A36', 'A36', 'A36', 'AISI 304', 'SS400', 'A36', 'SS400', 'AISI 316', 'AISI 316', 'A36', 'SS400', 'A36', 'AISI 316', 'A36', 'SS400', 'AISI 316', 'AISI 304', 'AISI 316', 'AISI 304', 'A36'],
    "抗變形能力": [154.86, 108.36, 161.06, 139.04, 188.47, 134.51, 116.22, 140.57, 110.49, 165.02, 108.7, 150.57, 108.44, 143.59, 182.87, 157.21, 132.73, 149.38, 150.36, 159.45],
    "摩擦係數": [0.245, 0.215, 0.23, 0.106, 0.224, 0.202, 0.209, 0.172, 0.116, 0.139, 0.29, 0.182, 0.193, 0.111, 0.108, 0.243, 0.208, 0.202, 0.149, 0.249],
    "軋輥直徑": [653.22, 640.43, 668.79, 647.6, 656.15, 601.3, 706.65, 645.07, 670.93, 624.92, 659.24, 610.91, 663.56, 639.32, 650.88, 670.94, 613.02, 629.47, 663.72, 712.95],
    "減少率": [0.36, 0.3, 0.36, 0.4, 0.33, 0.41, 0.46, 0.25, 0.34, 0.32, 0.41, 0.27, 0.39, 0.46, 0.33, 0.47, 0.5, 0.23, 0.27, 0.49],
    "應變率": [0.58, 0.8, 0.74, 0.66, 0.92, 0.87, 0.8, 0.94, 0.87, 0.72, 0.7, 0.88, 0.84, 0.82, 0.72, 0.86, 0.72, 0.79, 0.94, 0.73],
    "潤滑形式": ['oil', 'oil', 'water-based', 'dry', 'oil', 'water-based', 'dry', 'water-based', 'oil', 'water-based', 'water-based', 'water-based', 'water-based', 'dry', 'oil', 'dry', 'water-based', 'water-based', 'oil', 'oil'],
    "材料編號": [0, 0, 0, 1, 3, 0, 3, 2, 2, 0, 3, 0, 2, 0, 3, 2, 1, 2, 1, 0],
    "潤滑形式編號": [1, 1, 2, 0, 1, 2, 0, 2, 1, 2, 2, 2, 2, 0, 1, 0, 2, 2, 1, 1],
    "力矩": [1222.8, 1089.33, 1224.46, 1101.68, 1205.19, 1178.42, 1272.58, 1142.74, 1068.34, 1184.21, 1029.28, 1009.99, 1159.41, 1124.25, 1161.0, 1222.56, 1190.92, 1056.75, 1118.01, 1253.67]
}
df = pd.DataFrame(data)

# 為了讓範例可執行，處理可能缺失的類別
value_counts = df['材料分類'].value_counts()
to_remove = value_counts[value_counts < 2].index
if not to_remove.empty:
    df = df[~df['材料分類'].isin(to_remove)]

y = df['材料分類']
X = df.drop(['材料分類', '材料編號', '潤滑形式編號'], axis=1)

categorical_features = ['潤滑形式']
numerical_features = X.columns.drop(categorical_features).tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5)) # 確保 max_depth 與您圖片相似
])

model_pipeline.fit(X_train, y_train)
# --- 假設結束 ---

# 獲取所有特徵的名稱（包含One-Hot編碼後的）
feature_names_out = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
final_feature_names = feature_names_out.tolist()

# 視覺化決策樹
plt.figure(figsize=(28, 16)) # 調整圖形大小以容納更多內容

# 嘗試設定中文字體，優先使用 'SimHei' 或 'Microsoft JhengHei'
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft JhengHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題
    print("已嘗試設定中文字體為 'SimHei' 或 'Microsoft JhengHei'。")
except Exception as e:
    print(f"警告：設定中文字體失敗: {e}。圖表中的中文可能無法正常顯示。")
    print("請確保您的系統安裝了 'SimHei' 或 'Microsoft JhengHei' 字體。")
    print("若無，您可以嘗試安裝 'wqy-zenhei' (文泉驛正黑) 字體，並在程式碼中修改為 `plt.rcParams['font.sans-serif'] = ['wqy-zenhei']`。")

plot_tree(
    model_pipeline.named_steps['classifier'],
    feature_names=final_feature_names,
    class_names=sorted(y.unique().astype(str).tolist()),  # 確保 class_names 是字串且排序
    filled=True,
    rounded=True,
    fontsize=10, # 調整字體大小以適應圖形
    impurity=True,
    proportion=True,
    precision=2
)
plt.title("預測材料分類的決策樹", fontsize=20) # 設置標題為中文

plt.savefig('decision_tree_plot.png', dpi=300, bbox_inches='tight') # 儲存圖片
plt.show()

print("\n決策樹圖表已生成並儲存為 'decision_tree_plot.png'。")