import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pandas as pd
import os

# 獲取當前腳本的目錄
script_dir = os.path.dirname(__file__)

# 構建相對路徑
# 假設 '熱軋帶剛軋製資料.csv' 和 Python 腳本在同一個資料夾
file_name = '熱軋帶剛軋製資料.csv'
file_path = os.path.join(script_dir, file_name)

# 嘗試使用多種編碼讀取，提高成功率
try:
    df = pd.read_csv(file_path, encoding='utf-8')
    print("使用 UTF-8 編碼成功讀取檔案。")
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='big5')
        print("使用 Big5 編碼成功讀取檔案。")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='cp950') # cp950 是 Big5 的 Microsoft 擴展版本
            print("使用 CP950 編碼成功讀取檔案。")
        except UnicodeDecodeError:
            print(f"錯誤：無法使用已知編碼讀取檔案 '{file_path}'。請檢查檔案編碼。")
            df = None # 或者 raise the error again if you want the program to stop

if df is not None:
    print(df.head()) # 顯示 DataFrame 的前幾行以確認讀取成功

# 定義欄位名稱（請依照你的資料欄位名稱修改）
material_col = '材料分類'
lubrication_col = '潤滑形式編號'

# 先轉字串，再去除欄位開頭的數字與空白
df[material_col] = df[material_col].astype(str).str.replace(r'^\d+\s*', '', regex=True)
df[lubrication_col] = df[lubrication_col].astype(str).str.replace(r'^\d+\s*', '', regex=True)

# 設定繪圖風格與字體（避免中文亂碼）
sns.set(style="whitegrid")
plt.rcParams["font.family"] = "Arial Unicode MS"  # macOS用；Windows可換成"Microsoft JhengHei"或"SimHei"

# 假設「材料」與「潤滑形式」欄位名
material_col = '材料分類'
lubrication_col = '潤滑形式編號'

# 去除欄位中開頭的編號，例如 "01 鋼材A" 變成 "鋼材A"
df[material_col] = df[material_col].str.replace(r'^\d+\s*', '', regex=True)
df[lubrication_col] = df[lubrication_col].str.replace(r'^\d+\s*', '', regex=True)

# 取得數值欄位
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# 依「材料」分類
materials = df[material_col].unique()

for mat in materials:
    subset = df[df[material_col] == mat]

    fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(14, len(numeric_cols) * 3))
    fig.suptitle(f"材料分類：{mat}", fontsize=16)

    for i, col in enumerate(numeric_cols):
        sns.histplot(subset[col], kde=True, ax=axes[i, 0])
        axes[i, 0].set_title(f"{col} - 直方圖")
        axes[i, 0].set_xlabel(col)
        axes[i, 0].set_ylabel("頻率")

        sns.boxplot(x=subset[col], ax=axes[i, 1])
        axes[i, 1].set_title(f"{col} - 箱線圖")
        axes[i, 1].set_xlabel(col)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
