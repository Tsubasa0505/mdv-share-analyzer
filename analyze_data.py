import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（Windows用）
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

class MDVShareAnalyzer:
    def __init__(self):
        # Shift-JISエンコーディングでCSVを読み込み
        self.df = pd.read_csv('ダミーデータ（CSV）.csv', encoding='shift-jis')
        print(f"データ読み込み完了: {len(self.df)}行 × {len(self.df.columns)}列")
        print("\nカラム一覧:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"  {i}. {col}")
        
        # データ型の変換（パーセンテージを数値に）
        self._convert_percentage_columns()
        
    def _convert_percentage_columns(self):
        """パーセンテージ表記の列を数値に変換"""
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # %記号を含む列を数値に変換
                if self.df[col].astype(str).str.contains('%').any():
                    self.df[col] = self.df[col].str.replace('%', '').replace('-', '0').astype(float)
    
    def analyze_share_factors(self, target_col='製品Aシェア'):
        """製品シェアと各要因の関連性を分析"""
        print("\n" + "="*60)
        print(f"ターゲット変数: {target_col}")
        print("="*60)
        
        # 数値列のみを抽出
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # ターゲット列以外を説明変数として使用
        feature_cols = [col for col in numeric_cols if col != target_col and col in self.df.columns]
        
        if target_col not in self.df.columns:
            # 列名の候補を表示
            print(f"\n'{target_col}'列が見つかりません。")
            print("利用可能な数値列:")
            for col in numeric_cols:
                print(f"  - {col}")
            return
        
        # 1. 相関分析
        print("\n【1. 相関分析】")
        correlations = {}
        for col in feature_cols:
            if col in self.df.columns:
                valid_data = self.df[[target_col, col]].dropna()
                if len(valid_data) > 2:
                    corr, p_value = stats.pearsonr(valid_data[target_col], valid_data[col])
                    correlations[col] = {
                        '相関係数': corr,
                        'p値': p_value,
                        '有意性': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    }
        
        corr_df = pd.DataFrame(correlations).T
        corr_df = corr_df.sort_values('相関係数', key=abs, ascending=False)
        
        print("\n相関が強い変数TOP10:")
        print(corr_df.head(10))
        
        # 2. 重回帰分析
        print("\n【2. 重回帰分析】")
        
        # 有効なデータのみを使用
        valid_features = [col for col in feature_cols if col in self.df.columns]
        data_for_regression = self.df[[target_col] + valid_features].dropna()
        
        if len(data_for_regression) > 10 and len(valid_features) > 0:
            X = data_for_regression[valid_features]
            y = data_for_regression[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nモデルの決定係数 (R2): {r2:.4f}")
            
            # 回帰係数の表示
            coef_df = pd.DataFrame({
                '変数': valid_features,
                '回帰係数': model.coef_,
                '影響度（絶対値）': np.abs(model.coef_)
            }).sort_values('影響度（絶対値）', ascending=False)
            
            print("\n影響度の高い変数TOP10:")
            print(coef_df.head(10))
            
            # 3. 決定木分析
            print("\n【3. 決定木分析】")
            tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
            tree_model.fit(X_train, y_train)
            
            importance_df = pd.DataFrame({
                '変数': valid_features,
                '重要度': tree_model.feature_importances_
            }).sort_values('重要度', ascending=False)
            
            print("\n変数の重要度TOP10:")
            print(importance_df.head(10))
            
            # 可視化
            self._create_visualizations(target_col, corr_df, coef_df, importance_df, y_test, y_pred)
            
        else:
            print("回帰分析に必要なデータが不足しています")
    
    def _create_visualizations(self, target_col, corr_df, coef_df, importance_df, y_test, y_pred):
        """分析結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 相関係数の棒グラフ
        top_corr = corr_df.head(10)
        axes[0, 0].barh(range(len(top_corr)), top_corr['相関係数'].values)
        axes[0, 0].set_yticks(range(len(top_corr)))
        axes[0, 0].set_yticklabels(top_corr.index)
        axes[0, 0].set_xlabel('相関係数')
        axes[0, 0].set_title(f'{target_col}との相関（TOP10）')
        axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 2. 回帰係数の棒グラフ
        top_coef = coef_df.head(10)
        axes[0, 1].barh(range(len(top_coef)), top_coef['回帰係数'].values)
        axes[0, 1].set_yticks(range(len(top_coef)))
        axes[0, 1].set_yticklabels(top_coef['変数'].values)
        axes[0, 1].set_xlabel('回帰係数')
        axes[0, 1].set_title('重回帰分析：影響度（TOP10）')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 3. 変数重要度（決定木）
        top_importance = importance_df.head(10)
        axes[1, 0].barh(range(len(top_importance)), top_importance['重要度'].values)
        axes[1, 0].set_yticks(range(len(top_importance)))
        axes[1, 0].set_yticklabels(top_importance['変数'].values)
        axes[1, 0].set_xlabel('重要度')
        axes[1, 0].set_title('決定木分析：変数重要度（TOP10）')
        
        # 4. 予測精度の散布図
        axes[1, 1].scatter(y_test, y_pred, alpha=0.5)
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('実測値')
        axes[1, 1].set_ylabel('予測値')
        r2 = r2_score(y_test, y_pred)
        axes[1, 1].set_title(f'回帰モデルの予測精度 (R2={r2:.3f})')
        
        plt.tight_layout()
        plt.savefig('mdv_share_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("\n分析結果を 'mdv_share_analysis.png' に保存しました")
    
    def analyze_by_category(self, target_col='製品Aシェア', category_col='法人格'):
        """カテゴリ別の分析"""
        print("\n" + "="*60)
        print(f"カテゴリ別分析: {category_col}")
        print("="*60)
        
        if target_col not in self.df.columns or category_col not in self.df.columns:
            print(f"指定された列が見つかりません")
            return
        
        # カテゴリごとの統計
        category_stats = self.df.groupby(category_col)[target_col].agg(['mean', 'std', 'count'])
        category_stats = category_stats.sort_values('mean', ascending=False)
        
        print(f"\n{category_col}別の{target_col}統計:")
        print(category_stats)
        
        # 可視化
        fig, ax = plt.subplots(figsize=(10, 6))
        category_stats['mean'].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        ax.set_xlabel(category_col)
        ax.set_ylabel(f'{target_col}平均値')
        ax.set_title(f'{category_col}別の{target_col}分析')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('mdv_category_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("\nカテゴリ別分析結果を 'mdv_category_analysis.png' に保存しました")

def main():
    print("MDVデータ 製品シェア要因分析ツール")
    print("="*60)
    
    # 分析器を初期化
    analyzer = MDVShareAnalyzer()
    
    # メインの分析を実行
    analyzer.analyze_share_factors('製品Aシェア')
    
    # カテゴリ別分析
    analyzer.analyze_by_category('製品Aシェア', '法人格')
    
    print("\n分析完了！")

if __name__ == "__main__":
    main()