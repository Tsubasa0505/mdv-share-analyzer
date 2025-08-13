"""
MDV Share Analyzer 完全テストスクリプト
すべての機能をテストして動作確認を行います
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# テスト結果を記録
test_results = {
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "passed": 0,
    "failed": 0,
    "errors": []
}

def log_test(test_name, status, message=""):
    """テスト結果をログに記録"""
    result = {
        "test": test_name,
        "status": status,
        "message": message
    }
    test_results["tests"].append(result)
    if status == "PASS":
        test_results["passed"] += 1
        print(f"[PASS] {test_name}")
    else:
        test_results["failed"] += 1
        test_results["errors"].append(f"{test_name}: {message}")
        print(f"[FAIL] {test_name}: {message}")

def test_imports():
    """必要なライブラリのインポートテスト"""
    print("\n=== 1. ライブラリインポートテスト ===")
    try:
        import streamlit
        log_test("Streamlit import", "PASS")
    except ImportError as e:
        log_test("Streamlit import", "FAIL", str(e))
    
    try:
        import pandas
        log_test("Pandas import", "PASS")
    except ImportError as e:
        log_test("Pandas import", "FAIL", str(e))
    
    try:
        import numpy
        log_test("NumPy import", "PASS")
    except ImportError as e:
        log_test("NumPy import", "FAIL", str(e))
    
    try:
        import matplotlib
        log_test("Matplotlib import", "PASS")
    except ImportError as e:
        log_test("Matplotlib import", "FAIL", str(e))
    
    try:
        import seaborn
        log_test("Seaborn import", "PASS")
    except ImportError as e:
        log_test("Seaborn import", "FAIL", str(e))
    
    try:
        import plotly
        log_test("Plotly import", "PASS")
    except ImportError as e:
        log_test("Plotly import", "FAIL", str(e))
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        log_test("Scikit-learn import", "PASS")
    except ImportError as e:
        log_test("Scikit-learn import", "FAIL", str(e))
    
    try:
        import statsmodels.api
        log_test("Statsmodels import", "PASS")
    except ImportError as e:
        log_test("Statsmodels import", "FAIL", str(e))

def test_csv_operations():
    """CSVファイル操作のテスト"""
    print("\n=== 2. CSVファイル操作テスト ===")
    
    # テスト用CSVファイルを作成
    test_data = {
        '年月': pd.date_range('2024-01', periods=12, freq='M').strftime('%Y-%m'),
        'シェア': np.random.uniform(0.1, 0.5, 12),
        '売上': np.random.uniform(100000, 500000, 12),
        '広告費': np.random.uniform(10000, 50000, 12),
        '店舗数': np.random.randint(10, 50, 12),
        '従業員数': np.random.randint(50, 200, 12),
        '顧客満足度': np.random.uniform(3.0, 5.0, 12)
    }
    
    df = pd.DataFrame(test_data)
    test_csv_path = "test_data.csv"
    
    try:
        # CSV書き込みテスト
        df.to_csv(test_csv_path, index=False, encoding='utf-8-sig')
        log_test("CSV write", "PASS")
        
        # CSV読み込みテスト
        df_read = pd.read_csv(test_csv_path, encoding='utf-8-sig')
        if len(df_read) == len(df):
            log_test("CSV read", "PASS")
        else:
            log_test("CSV read", "FAIL", "データ行数が一致しません")
        
        # データ型チェック
        if df_read['シェア'].dtype == np.float64:
            log_test("Data type check", "PASS")
        else:
            log_test("Data type check", "FAIL", "データ型が正しくありません")
            
    except Exception as e:
        log_test("CSV operations", "FAIL", str(e))
    
    return test_csv_path

def test_analysis_functions():
    """分析関数のテスト"""
    print("\n=== 3. 分析関数テスト ===")
    
    # テストデータ作成
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    y = 2 * X['feature1'] + 1.5 * X['feature2'] + np.random.randn(n_samples) * 0.1
    
    try:
        # 相関分析テスト
        corr_matrix = X.corr()
        if corr_matrix.shape == (3, 3):
            log_test("Correlation analysis", "PASS")
        else:
            log_test("Correlation analysis", "FAIL", "相関行列のサイズが正しくありません")
    except Exception as e:
        log_test("Correlation analysis", "FAIL", str(e))
    
    try:
        # 回帰分析テスト
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        if 0 <= score <= 1:
            log_test("Regression analysis", "PASS")
        else:
            log_test("Regression analysis", "FAIL", f"R2スコアが異常です: {score}")
    except Exception as e:
        log_test("Regression analysis", "FAIL", str(e))
    
    try:
        # 決定木分析テスト
        from sklearn.tree import DecisionTreeRegressor
        
        tree_model = DecisionTreeRegressor(random_state=42, max_depth=3)
        tree_model.fit(X_train, y_train)
        importance = tree_model.feature_importances_
        
        if len(importance) == 3 and abs(sum(importance) - 1.0) < 0.001:
            log_test("Decision tree analysis", "PASS")
        else:
            log_test("Decision tree analysis", "FAIL", "特徴量重要度の計算が正しくありません")
    except Exception as e:
        log_test("Decision tree analysis", "FAIL", str(e))
    
    try:
        # PCA分析テスト
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        if X_pca.shape == (n_samples, 2):
            log_test("PCA analysis", "PASS")
        else:
            log_test("PCA analysis", "FAIL", "PCA変換後の次元が正しくありません")
    except Exception as e:
        log_test("PCA analysis", "FAIL", str(e))

def test_statistical_validations():
    """統計的検証機能のテスト"""
    print("\n=== 4. 統計的検証テスト ===")
    
    # テストデータ
    np.random.seed(42)
    data = np.random.randn(100)
    
    try:
        # 正規性検定
        from scipy import stats
        statistic, p_value = stats.shapiro(data)
        if 0 <= p_value <= 1:
            log_test("Shapiro-Wilk test", "PASS")
        else:
            log_test("Shapiro-Wilk test", "FAIL", f"p値が範囲外: {p_value}")
    except Exception as e:
        log_test("Shapiro-Wilk test", "FAIL", str(e))
    
    try:
        # VIF計算テスト
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = pd.DataFrame({
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'x3': np.random.randn(100)
        })
        vif = variance_inflation_factor(X.values, 0)
        if vif > 0:
            log_test("VIF calculation", "PASS")
        else:
            log_test("VIF calculation", "FAIL", f"VIF値が異常: {vif}")
    except Exception as e:
        log_test("VIF calculation", "FAIL", str(e))
    
    try:
        # 外れ値検出テスト
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        outliers = np.sum((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR))
        log_test("Outlier detection", "PASS", f"{outliers}個の外れ値を検出")
    except Exception as e:
        log_test("Outlier detection", "FAIL", str(e))

def test_visualization():
    """可視化機能のテスト"""
    print("\n=== 5. 可視化機能テスト ===")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # テストプロット作成
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close()
        log_test("Matplotlib plot", "PASS")
    except Exception as e:
        log_test("Matplotlib plot", "FAIL", str(e))
    
    try:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        log_test("Plotly plot", "PASS")
    except Exception as e:
        log_test("Plotly plot", "FAIL", str(e))

def test_app_syntax():
    """app.pyの構文チェック"""
    print("\n=== 6. アプリケーション構文チェック ===")
    
    try:
        import ast
        with open('app.py', 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        log_test("app.py syntax", "PASS")
    except SyntaxError as e:
        log_test("app.py syntax", "FAIL", f"Line {e.lineno}: {e.msg}")
    except Exception as e:
        log_test("app.py syntax", "FAIL", str(e))

def test_what_if_simulation():
    """What-Ifシミュレーション機能のテスト"""
    print("\n=== 7. What-Ifシミュレーションテスト ===")
    
    try:
        # モデルのモック作成
        from sklearn.linear_model import LinearRegression
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        model = LinearRegression()
        model.fit(X, y)
        
        # 予測テスト
        X_test = np.random.randn(1, 3)
        prediction = model.predict(X_test)
        
        if isinstance(prediction[0], (int, float)):
            log_test("What-If prediction", "PASS")
        else:
            log_test("What-If prediction", "FAIL", "予測値の型が正しくありません")
            
        # 感度分析テスト
        base_prediction = model.predict(X_test)[0]
        sensitivity = {}
        for i in range(3):
            X_modified = X_test.copy()
            X_modified[0, i] *= 1.1  # 10%増加
            new_prediction = model.predict(X_modified)[0]
            sensitivity[f'feature_{i}'] = (new_prediction - base_prediction) / base_prediction
        
        if len(sensitivity) == 3:
            log_test("Sensitivity analysis", "PASS")
        else:
            log_test("Sensitivity analysis", "FAIL", "感度分析の結果が正しくありません")
            
    except Exception as e:
        log_test("What-If simulation", "FAIL", str(e))

def test_prompt_generation():
    """AIプロンプト生成機能のテスト"""
    print("\n=== 8. AIプロンプト生成テスト ===")
    
    try:
        # テスト用の分析結果
        analysis_summary = {
            "基本統計": {
                "サンプル数": 100,
                "平均": 0.5,
                "標準偏差": 0.1
            },
            "相関分析": {
                "最大相関": 0.8,
                "最小相関": 0.1
            }
        }
        
        # プロンプト生成
        import json
        analysis_data_str = json.dumps(analysis_summary, ensure_ascii=False, indent=2)
        
        prompt_template = f"""
以下のデータ分析結果を解釈して、分かりやすく説明してください。

## 分析結果
{analysis_data_str}

## 解釈してほしいポイント
1. この分析結果から何が分かるか
2. 実務への活用方法
"""
        
        if len(prompt_template) > 100 and "分析結果" in prompt_template:
            log_test("Prompt generation", "PASS", f"プロンプト長: {len(prompt_template)}文字")
        else:
            log_test("Prompt generation", "FAIL", "プロンプトが正しく生成されていません")
            
    except Exception as e:
        log_test("Prompt generation", "FAIL", str(e))

def cleanup_test_files():
    """テストファイルのクリーンアップ"""
    print("\n=== 9. クリーンアップ ===")
    
    try:
        if os.path.exists('test_data.csv'):
            os.remove('test_data.csv')
            log_test("Cleanup test files", "PASS")
        else:
            log_test("Cleanup test files", "PASS", "削除するファイルなし")
    except Exception as e:
        log_test("Cleanup test files", "FAIL", str(e))

def main():
    """メインテスト実行"""
    print("=" * 50)
    print("MDV Share Analyzer 完全テスト")
    print("=" * 50)
    
    # 各テストを実行
    test_imports()
    test_csv_path = test_csv_operations()
    test_analysis_functions()
    test_statistical_validations()
    test_visualization()
    test_app_syntax()
    test_what_if_simulation()
    test_prompt_generation()
    cleanup_test_files()
    
    # 結果サマリー
    print("\n" + "=" * 50)
    print("テスト結果サマリー")
    print("=" * 50)
    print(f"成功: {test_results['passed']}件")
    print(f"失敗: {test_results['failed']}件")
    print(f"合計: {test_results['passed'] + test_results['failed']}件")
    
    if test_results['failed'] > 0:
        print("\n失敗したテスト:")
        for error in test_results['errors']:
            print(f"  - {error}")
    
    # 結果をJSONファイルに保存
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print(f"\n詳細な結果は test_results.json に保存されました")
    
    # 終了コード
    return 0 if test_results['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())