import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ---[ 步驟 1-6: 與前次程式碼相同，建立並訓練模型 ]---
file_path = '熱.txt'
try:
    df = pd.read_csv(file_path, sep='\t', on_bad_lines='skip')
    df.dropna(how='all', inplace=True)

    y = df['材料分類']
    X = df.drop(columns=['材料分類', '材料編號', '潤滑形式編號'])

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', LogisticRegression(random_state=42, max_iter=1000))])

    X_train, X_test, y_train,y_test_data = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)
    print("模型訓練完成。\n")

    # ---[ 步驟 7: 產生羅吉斯迴歸相關圖形 ]---

    # 圖形 1: ROC 曲線 (適用於多類別分類)
    # ------------------------------------
    y_score = model.decision_function(X_test)
    y_test_binarized = label_binarize(y_test_data, classes=model.classes_)
    n_classes = len(model.classes_)

    # 計算每個類別的 ROC 曲線
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # 繪製所有類別的 ROC 曲線
    plt.figure(figsize=(10, 8))
    colors = ['aqua', 'darkorange', 'cornflowerblue', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(model.classes_[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('假陽性率 (False Positive Rate)')
    plt.ylabel('真陽性率 (True Positive Rate)')
    plt.title('多類別 ROC 曲線 (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.show()

    # 圖形 2: 特徵係數圖 (視覺化特徵重要性)
    # ------------------------------------
    # 從 Pipeline 中獲取訓練好的模型和特徵名稱
    final_classifier = model.named_steps['classifier']
    final_preprocessor = model.named_steps['preprocessor']

    # 獲取 one-hot encoding 後的類別特徵名稱
    ohe_feature_names = final_preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)

    # 組合所有特徵的名稱
    all_feature_names = np.concatenate([numerical_features, ohe_feature_names])

    # 建立一個 DataFrame 來儲存特徵和它們的係數
    # 由於是多類別分類，每個類別都有一組係數
    coef_df = pd.DataFrame(final_classifier.coef_, columns=all_feature_names, index=model.classes_)

    # 繪製每個類別的特徵係數
    fig, axes = plt.subplots(n_classes, 1, figsize=(12, 18), sharex=True)
    fig.suptitle('各材料分類的特徵係數', fontsize=16)

    for i, cls in enumerate(model.classes_):
        sorted_coef = coef_df.loc[cls].sort_values(ascending=False)
        sns.barplot(x=sorted_coef.values, y=sorted_coef.index, ax=axes[i], palette='viridis')
        axes[i].set_title(f'Class: {cls}')
        axes[i].tick_params(axis='y', labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


except FileNotFoundError:
    print(f"錯誤：找不到檔案 '{file_path}'。請確保檔案與程式碼在同一個目錄下。")
except Exception as e:
    print(f"發生錯誤：{e}")