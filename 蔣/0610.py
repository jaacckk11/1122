import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import os


def load_and_prepare_data(file_path):
    """
    從 CSV 載入數據，將欄位重新命名為英文以便處理，
    並在適當的情況下將欄位轉換為數值型態。
    """
    if not os.path.exists(file_path):
        print(f"錯誤：在指定路徑找不到檔案 '{file_path}'。")
        print("請確認路徑和檔名是否完全正確。")
        return None

    df = None
    for enc in ['utf-8', 'big5', 'gbk']:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"成功使用編碼 {enc} 載入 CSV 檔案。")
            break
        except Exception:
            continue
    if df is None:
        print("載入 CSV 失敗。")
        return None

    column_mapping = {
        '入口溫度': 'InletTemp', '出口溫度': 'OutletTemp', '滾軋速度': 'RollingSpeed',
        '帶條厚度': 'StripThickness', '材料分類': 'MaterialGrade', '抗變形能力': 'DeformationResistance',
        '摩擦係數': 'FrictionCoeff', '軋輥直徑': 'RollDiameter', '減少率': 'ReductionRate',
        '應變率': 'StrainRate', '潤滑形式': 'LubricationType', '材料編號': 'MaterialID',
        '潤滑形式編號': 'LubricationFormID', '力矩': 'Torque'
    }
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

    potential_numeric_cols = list(column_mapping.values())
    for col in potential_numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def plot_polynomial_regression(df, x_col, y_col, x_label, y_label, plot_title, filename, degree=2):
    """
    繪製兩個變數的散佈圖，並疊加多項式迴歸擬合曲線。
    """
    if df is None or x_col not in df.columns or y_col not in df.columns:
        print(f"錯誤：DataFrame 中缺少 '{x_col}' 或 '{y_col}' 欄位。")
        return

    # 準備繪圖數據，移除這兩個欄位中的遺失值
    plot_data = df[[x_col, y_col]].dropna()
    if plot_data.empty:
        print(f"警告：移除遺失值後，'{x_col}' 和 '{y_col}' 之間沒有可用的數據進行繪圖。")
        return

    X = plot_data[[x_col]]
    y = plot_data[y_col]

    # 建立並擬合多項式迴歸模型
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    model.fit(X, y)

    # 產生用於繪製擬合曲線的平滑數據點
    x_range = np.linspace(X.min(), X.max(), 100)
    y_pred_range = model.predict(x_range)

    # 繪圖
    plt.figure(figsize=(10, 7))
    plt.scatter(X, y, alpha=0.6, label='實際數據點')
    plt.plot(x_range, y_pred_range, color='red', linewidth=2, label=f'{degree}次多項式迴歸擬合線')

    plt.title(plot_title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    try:
        # 確保儲存圖形的目錄存在
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(filename)
        print(f"圖表已儲存為：{filename}")
    except Exception as e:
        print(f"儲存圖表 '{filename}' 時發生錯誤：{e}")
    plt.close()


# --- 主程式執行流程 ---

# 檔案路徑設定：
# 您可以選擇使用相對路徑或絕對路徑。
# 1. 相對路徑：將 '熱軋帶剛軋製資料.csv' 檔案放在與此 Python 腳本相同的資料夾中。
#    csv_file_path = "熱軋帶剛軋製資料.csv"
# 2. 絕對路徑：提供檔案在您電腦上的完整路徑。
#    為了避免路徑中的反斜線 `\` 產生問題，建議在字串前加上 r (raw string)，如下所示。
#    請確認您的 CSV 檔案確實存在於此路徑下。
csv_file_path = r"C:\Users\jackj\Desktop\彰師作業\進階程式設計\小組報告\熱軋帶剛軋製資料.csv"
output_directory = r"C:\Users\jackj\Desktop\彰師作業\進階程式設計\小組報告"

data_df = load_and_prepare_data(csv_file_path)

if data_df is not None:
    # 設定中文字體
    try:
        plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體
        plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號
        print("\n已嘗試設定中文字體 'Microsoft JhengHei'。")
    except Exception as e:
        print(f"\n無法設定中文字體 'Microsoft JhengHei' ({e})。圖表標籤可能無法正確顯示中文。")

    print("\n--- 開始生成圖形 ---")

    # 圖一：減少率 vs. 力矩
    plot_polynomial_regression(
        df=data_df,
        x_col='ReductionRate',
        y_col='Torque',
        x_label='減少率',
        y_label='力矩',
        plot_title='減少率 與 力矩 的關係 (二次多項式迴歸)',
        filename=os.path.join(output_directory, 'plot_reduction_vs_torque.png')
    )

    # 圖二：減少率 vs. 抗變形能力
    plot_polynomial_regression(
        df=data_df,
        x_col='ReductionRate',
        y_col='DeformationResistance',
        x_label='減少率',
        y_label='抗變形能力',
        plot_title='減少率 與 抗變形能力 的關係 (二次多項式迴歸)',
        filename=os.path.join(output_directory, 'plot_reduction_vs_deformation_resistance.png')
    )

    # 圖三：入口溫度 vs. 減少率
    plot_polynomial_regression(
        df=data_df,
        x_col='InletTemp',
        y_col='ReductionRate',
        x_label='入口溫度',
        y_label='減少率',
        plot_title='入口溫度 與 減少率 的關係 (二次多項式迴歸)',
        filename=os.path.join(output_directory, 'plot_inlettemp_vs_reduction.png')
    )

    print("\n所有圖形已生成完畢。")
else:
    print("\n數據載入失敗，無法生成圖形。")

