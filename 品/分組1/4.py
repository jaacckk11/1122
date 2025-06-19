import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

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

material_col = '材料分類'
green_reduction_col = '減少率'

rolling_speed_col = '摩擦係數'  # 請確認你的欄位名稱是否正確

materials = df[material_col].unique()

# 摩擦係數 vs 減少綠（三圖）
fig_speed, axs_speed = plt.subplots(1, 3, figsize=(18, 5))
for ax, material in zip(axs_speed, materials):
    subset = df[df[material_col] == material]
    ax.scatter(subset[rolling_speed_col], subset[green_reduction_col], color='tab:green')
    ax.set_title(f'{material}：摩擦係數 vs 減少率')
    ax.set_xlabel('摩擦係數')
    ax.set_ylabel('減少率 (%)')
    ax.grid(True)
plt.tight_layout()
plt.show()
