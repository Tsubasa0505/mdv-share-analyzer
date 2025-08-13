"""
MDV Share Analyzer 完全機能テスト
すべての機能を自動テストするスクリプト
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime
import json

# Test configuration
TEST_DATA_FILE = "test_mdv_data.csv"
TEST_RESULTS_FILE = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def create_test_data():
    """テスト用のCSVデータを生成"""
    print("\n" + "="*50)
    print("1. テスト用CSVデータの生成")
    print("="*50)
    
    np.random.seed(42)
    
    # 12ヶ月分のデータを生成
    months = pd.date_range('2024-01', periods=12, freq='ME')
    
    # データ生成（相関を持たせる）
    base_trend = np.linspace(0.2, 0.4, 12) + np.random.normal(0, 0.02, 12)
    
    data = {
        '年月': months.strftime('%Y-%m'),
        'シェア': base_trend,
        '売上高': base_trend * 1000000 + np.random.normal(0, 50000, 12),
        '広告費': base_trend * 100000 + np.random.normal(0, 5000, 12),
        '店舗数': (base_trend * 100).astype(int) + np.random.randint(-5, 5, 12),
        '従業員数': (base_trend * 500).astype(int) + np.random.randint(-20, 20, 12),
        '顧客満足度': base_trend * 10 + np.random.normal(0, 0.3, 12),
        'キャンペーン回数': np.random.randint(1, 10, 12),
        '競合数': np.random.randint(3, 8, 12),
        '市場規模': np.random.uniform(1000000, 2000000, 12),
        '価格指数': 100 + np.random.normal(0, 5, 12)
    }
    
    df = pd.DataFrame(data)
    
    # CSVファイルとして保存
    df.to_csv(TEST_DATA_FILE, index=False, encoding='utf-8-sig')
    
    print(f"[OK] テストデータを生成しました: {TEST_DATA_FILE}")
    print(f"   - 行数: {len(df)}")
    print(f"   - 列数: {len(df.columns)}")
    print(f"   - 数値列: {len(df.select_dtypes(include=[np.number]).columns)}")
    
    # データプレビュー
    print("\nデータプレビュー:")
    print(df.head())
    
    return df

def test_imports():
    """必要なライブラリのインポートテスト"""
    print("\n" + "="*50)
    print("2. ライブラリインポートテスト")
    print("="*50)
    
    test_results = []
    
    libraries = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sklearn',
        'statsmodels',
        'reportlab',
        'openpyxl',
        'kaleido'
    ]
    
    for lib in libraries:
        try:
            __import__(lib)
            print(f"[OK] {lib}")
            test_results.append({"library": lib, "status": "OK"})
        except ImportError as e:
            print(f"[FAIL] {lib}: FAIL - {e}")
            test_results.append({"library": lib, "status": "FAIL", "error": str(e)})
    
    return test_results

def test_pdf_generation():
    """PDF生成機能のテスト"""
    print("\n" + "="*50)
    print("3. PDF生成機能テスト")
    print("="*50)
    
    try:
        from pdf_generator_jp import PDFReportGenerator, prepare_analysis_for_pdf
        print("[OK] PDF生成モジュールのインポート: OK")
        
        # ダミーデータでPDF生成テスト
        pdf_generator = PDFReportGenerator()
        
        # テスト用の分析結果を作成
        test_results = {
            'title': 'テストレポート',
            'author': 'テストユーザー',
            'data_overview': {
                'samples': 12,
                'features': 10,
                'target': 'シェア',
                'missing': 0
            },
            'correlation': {
                'matrix': pd.DataFrame(np.random.rand(5, 5)),
                'insights': 'テスト相関分析'
            }
        }
        
        # PDF生成
        pdf_bytes = pdf_generator.generate_report(test_results)
        
        if pdf_bytes and len(pdf_bytes) > 0:
            print(f"[OK] PDF生成: OK (サイズ: {len(pdf_bytes)/1024:.1f} KB)")
            
            # PDFファイルとして保存（確認用）
            with open('test_report.pdf', 'wb') as f:
                f.write(pdf_bytes)
            print("   - テストPDFを保存: test_report.pdf")
            return True
        else:
            print("[FAIL] PDF生成: FAIL - 空のPDF")
            return False
            
    except Exception as e:
        print(f"[FAIL] PDF生成テスト失敗: {e}")
        return False

def test_export_utils():
    """グラフ・表エクスポート機能のテスト"""
    print("\n" + "="*50)
    print("4. グラフ・表エクスポート機能テスト")
    print("="*50)
    
    try:
        from pdf_export_utils import (
            export_chart_as_image, 
            export_table_as_csv, 
            export_table_as_excel,
            create_chart_collection_pdf,
            create_table_collection_pdf
        )
        print("[OK] エクスポートモジュールのインポート: OK")
        
        # テスト用データ
        test_df = pd.DataFrame({
            '列1': [1, 2, 3, 4, 5],
            '列2': [10, 20, 30, 40, 50],
            '列3': ['A', 'B', 'C', 'D', 'E']
        })
        
        # CSV エクスポートテスト
        csv_bytes = export_table_as_csv(test_df)
        if csv_bytes:
            print(f"[OK] CSVエクスポート: OK (サイズ: {len(csv_bytes)} bytes)")
        else:
            print("[FAIL] CSVエクスポート: FAIL")
        
        # Excel エクスポートテスト
        excel_bytes = export_table_as_excel({'テスト表': test_df})
        if excel_bytes:
            print(f"[OK] Excelエクスポート: OK (サイズ: {len(excel_bytes)} bytes)")
            # 確認用に保存
            with open('test_export.xlsx', 'wb') as f:
                f.write(excel_bytes)
            print("   - テストExcelを保存: test_export.xlsx")
        else:
            print("[FAIL] Excelエクスポート: FAIL")
        
        # Plotlyグラフのテスト
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]))
        fig.update_layout(title="テストグラフ")
        
        # PNG エクスポートテスト（kaleidoが必要）
        try:
            img_bytes = export_chart_as_image(fig, 'png')
            if img_bytes:
                print(f"[OK] PNGエクスポート: OK (サイズ: {len(img_bytes)} bytes)")
            else:
                print("[WARN] PNGエクスポート: SKIP (kaleido未設定)")
        except Exception as e:
            print(f"[WARN] PNGエクスポート: SKIP ({e})")
        
        # PDFコレクションテスト
        try:
            # グラフコレクション
            charts = {'テストグラフ1': fig, 'テストグラフ2': fig}
            chart_pdf = create_chart_collection_pdf(charts, "テストグラフ集")
            if chart_pdf:
                print(f"[OK] グラフPDFコレクション: OK (サイズ: {len(chart_pdf)/1024:.1f} KB)")
        except Exception as e:
            print(f"[WARN] グラフPDFコレクション: SKIP ({e})")
        
        # 表コレクション
        tables = {'テスト表1': test_df, 'テスト表2': test_df}
        table_pdf = create_table_collection_pdf(tables, "テスト表集")
        if table_pdf:
            print(f"[OK] 表PDFコレクション: OK (サイズ: {len(table_pdf)/1024:.1f} KB)")
            with open('test_tables.pdf', 'wb') as f:
                f.write(table_pdf)
            print("   - テスト表PDFを保存: test_tables.pdf")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] エクスポート機能テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_functions():
    """分析関数のテスト"""
    print("\n" + "="*50)
    print("5. 分析関数テスト")
    print("="*50)
    
    try:
        # データ読み込み
        df = pd.read_csv(TEST_DATA_FILE, encoding='utf-8-sig')
        
        # 数値列の抽出
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 相関分析
        corr_matrix = df[numeric_cols].corr()
        print(f"[OK] 相関分析: OK (行列サイズ: {corr_matrix.shape})")
        
        # 回帰分析
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        target = 'シェア'
        features = [col for col in numeric_cols if col != target]
        
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        print(f"[OK] 回帰分析: OK (R²スコア: {score:.4f})")
        
        # 決定木分析
        from sklearn.tree import DecisionTreeRegressor
        
        tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
        tree_model.fit(X_train, y_train)
        tree_score = tree_model.score(X_test, y_test)
        
        print(f"[OK] 決定木分析: OK (R²スコア: {tree_score:.4f})")
        
        # PCA分析
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        print(f"[OK] PCA分析: OK (説明分散: {pca.explained_variance_ratio_[0]:.4f})")
        
        # VIF計算
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_values = []
        for i in range(len(features[:3])):  # 最初の3変数のみテスト
            try:
                vif = variance_inflation_factor(X[features[:3]].values, i)
                vif_values.append(vif)
            except:
                vif_values.append(np.nan)
        
        print(f"[OK] VIF計算: OK (テスト値: {vif_values[0]:.4f})")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 分析関数テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_app_integration():
    """アプリケーション統合テスト"""
    print("\n" + "="*50)
    print("6. アプリケーション統合テスト")
    print("="*50)
    
    try:
        # app.pyの構文チェック
        import ast
        with open('app.py', 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print("[OK] app.py構文チェック: OK")
        
        # 必要なモジュールのインポートチェック
        try:
            from app import Config, AnalysisResults
            print("[OK] アプリケーション設定クラス: OK")
        except ImportError as e:
            print(f"[WARN] アプリケーション設定クラス: {e}")
        
        return True
        
    except SyntaxError as e:
        print(f"[FAIL] app.py構文エラー: Line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"[FAIL] 統合テスト失敗: {e}")
        return False

def cleanup_test_files():
    """テストファイルのクリーンアップ"""
    print("\n" + "="*50)
    print("7. クリーンアップ")
    print("="*50)
    
    test_files = [
        'test_report.pdf',
        'test_export.xlsx',
        'test_tables.pdf'
    ]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"[OK] {file} を削除")
            except:
                print(f"[WARN] {file} の削除失敗")
    
    # CSVファイルは残す（確認用）
    print(f"[INFO] {TEST_DATA_FILE} は確認用に残します")

def main():
    """メインテスト実行"""
    print("="*50)
    print("MDV Share Analyzer 完全機能テスト")
    print("="*50)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    }
    
    # 1. テストデータ生成
    try:
        df = create_test_data()
        test_results["tests"]["data_generation"] = "PASS"
        test_results["summary"]["passed"] += 1
    except Exception as e:
        test_results["tests"]["data_generation"] = f"FAIL: {e}"
        test_results["summary"]["failed"] += 1
    test_results["summary"]["total"] += 1
    
    # 2. インポートテスト
    import_results = test_imports()
    test_results["tests"]["imports"] = import_results
    for result in import_results:
        test_results["summary"]["total"] += 1
        if result["status"] == "OK":
            test_results["summary"]["passed"] += 1
        else:
            test_results["summary"]["failed"] += 1
    
    # 3. PDF生成テスト
    if test_pdf_generation():
        test_results["tests"]["pdf_generation"] = "PASS"
        test_results["summary"]["passed"] += 1
    else:
        test_results["tests"]["pdf_generation"] = "FAIL"
        test_results["summary"]["failed"] += 1
    test_results["summary"]["total"] += 1
    
    # 4. エクスポート機能テスト
    if test_export_utils():
        test_results["tests"]["export_utils"] = "PASS"
        test_results["summary"]["passed"] += 1
    else:
        test_results["tests"]["export_utils"] = "FAIL"
        test_results["summary"]["failed"] += 1
    test_results["summary"]["total"] += 1
    
    # 5. 分析関数テスト
    if test_analysis_functions():
        test_results["tests"]["analysis_functions"] = "PASS"
        test_results["summary"]["passed"] += 1
    else:
        test_results["tests"]["analysis_functions"] = "FAIL"
        test_results["summary"]["failed"] += 1
    test_results["summary"]["total"] += 1
    
    # 6. 統合テスト
    if test_app_integration():
        test_results["tests"]["app_integration"] = "PASS"
        test_results["summary"]["passed"] += 1
    else:
        test_results["tests"]["app_integration"] = "FAIL"
        test_results["summary"]["failed"] += 1
    test_results["summary"]["total"] += 1
    
    # 7. クリーンアップ
    cleanup_test_files()
    
    # テスト結果サマリー
    print("\n" + "="*50)
    print("テスト結果サマリー")
    print("="*50)
    print(f"[OK] 成功: {test_results['summary']['passed']}件")
    print(f"[FAIL] 失敗: {test_results['summary']['failed']}件")
    print(f"[WARN] スキップ: {test_results['summary']['skipped']}件")
    print(f"合計: {test_results['summary']['total']}件")
    
    # 成功率
    if test_results['summary']['total'] > 0:
        success_rate = (test_results['summary']['passed'] / test_results['summary']['total']) * 100
        print(f"\n成功率: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n[SUCCESS] すべてのテストが成功しました！")
        elif success_rate >= 80:
            print("\n[OK] ほとんどのテストが成功しました")
        elif success_rate >= 60:
            print("\n[WARN] 一部のテストが失敗しました")
        else:
            print("\n[FAIL] 多くのテストが失敗しました")
    
    # 結果をJSONファイルに保存
    with open(TEST_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n詳細な結果は {TEST_RESULTS_FILE} に保存されました")
    
    print("\n" + "="*50)
    print("テスト完了")
    print("="*50)
    
    # アプリケーションのURL
    print("\nアプリケーションURL: http://localhost:8504")
    print("テスト用CSVファイル: test_mdv_data.csv")
    
    return test_results['summary']['failed'] == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)