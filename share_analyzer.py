import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

class ShareAnalyzer:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"データ読み込み完了: {len(self.df)}行 × {len(self.df.columns)}列")
        print(f"カラム: {', '.join(self.df.columns.tolist())}")
        self.target_col = None
        self.feature_cols = []
        self.results = {}
        
    def set_target_and_features(self, target_col, feature_cols=None):
        if target_col not in self.df.columns:
            raise ValueError(f"ターゲット列 '{target_col}' が見つかりません")
        
        self.target_col = target_col
        
        if feature_cols is None:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_cols = [col for col in numeric_cols if col != target_col]
        else:
            self.feature_cols = feature_cols
            
        print(f"\nターゲット変数: {self.target_col}")
        print(f"説明変数: {', '.join(self.feature_cols)}")
        
        missing_cols = self.df[self.feature_cols + [self.target_col]].isnull().sum()
        if missing_cols.any():
            print("\n欠損値の状況:")
            print(missing_cols[missing_cols > 0])
            
    def basic_statistics(self):
        print("\n" + "="*50)
        print("基本統計量")
        print("="*50)
        
        cols_to_analyze = [self.target_col] + self.feature_cols
        stats_df = self.df[cols_to_analyze].describe()
        print(stats_df)
        
        self.results['basic_stats'] = stats_df
        return stats_df
    
    def correlation_analysis(self):
        print("\n" + "="*50)
        print("相関分析")
        print("="*50)
        
        correlations = {}
        for col in self.feature_cols:
            valid_data = self.df[[self.target_col, col]].dropna()
            
            pearson_r, pearson_p = stats.pearsonr(valid_data[self.target_col], valid_data[col])
            spearman_r, spearman_p = stats.spearmanr(valid_data[self.target_col], valid_data[col])
            
            correlations[col] = {
                'Pearson相関係数': pearson_r,
                'Pearson_p値': pearson_p,
                'Spearman相関係数': spearman_r,
                'Spearman_p値': spearman_p,
                '有意性(p<0.05)': 'あり' if pearson_p < 0.05 else 'なし'
            }
        
        corr_df = pd.DataFrame(correlations).T
        corr_df = corr_df.sort_values('Pearson相関係数', ascending=False)
        
        print("\n相関係数（降順）:")
        print(corr_df)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        corr_matrix = self.df[[self.target_col] + self.feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0], fmt='.2f', cbar_kws={'label': '相関係数'})
        axes[0].set_title('相関行列ヒートマップ')
        
        important_vars = corr_df[corr_df['有意性(p<0.05)'] == 'あり'].index.tolist()[:5]
        if important_vars:
            axes[1].barh(range(len(important_vars)), 
                        [corr_df.loc[var, 'Pearson相関係数'] for var in important_vars])
            axes[1].set_yticks(range(len(important_vars)))
            axes[1].set_yticklabels(important_vars)
            axes[1].set_xlabel('Pearson相関係数')
            axes[1].set_title(f'{self.target_col}と有意な相関を持つ上位変数')
            axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            for i, var in enumerate(important_vars):
                val = corr_df.loc[var, 'Pearson相関係数']
                axes[1].text(val, i, f'{val:.3f}', va='center', 
                           ha='left' if val > 0 else 'right')
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        self.results['correlations'] = corr_df
        return corr_df
    
    def regression_analysis(self):
        print("\n" + "="*50)
        print("回帰分析")
        print("="*50)
        
        data = self.df[[self.target_col] + self.feature_cols].dropna()
        X = data[self.feature_cols]
        y = data[self.target_col]
        
        print("\n単回帰分析:")
        single_reg_results = {}
        for col in self.feature_cols:
            X_single = data[[col]]
            model = LinearRegression()
            model.fit(X_single, y)
            y_pred = model.predict(X_single)
            r2 = r2_score(y, y_pred)
            
            single_reg_results[col] = {
                '回帰係数': model.coef_[0],
                '切片': model.intercept_,
                'R²': r2
            }
        
        single_reg_df = pd.DataFrame(single_reg_results).T
        single_reg_df = single_reg_df.sort_values('R²', ascending=False)
        print(single_reg_df)
        
        print("\n重回帰分析:")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"\n重回帰モデルの性能:")
        print(f"訓練データ R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
        print(f"テストデータ R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        print("\n回帰係数:")
        coef_df = pd.DataFrame({
            '変数': self.feature_cols,
            '回帰係数': model.coef_,
            '影響度(絶対値)': np.abs(model.coef_)
        }).sort_values('影響度(絶対値)', ascending=False)
        print(coef_df)
        print(f"切片: {model.intercept_:.4f}")
        
        X_sm = sm.add_constant(X)
        model_sm = sm.OLS(y, X_sm).fit()
        
        print("\n統計的検定結果:")
        summary_data = {
            '変数': ['定数項'] + self.feature_cols,
            '係数': model_sm.params.values,
            '標準誤差': model_sm.bse.values,
            't値': model_sm.tvalues.values,
            'p値': model_sm.pvalues.values,
            '有意性': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' 
                    for p in model_sm.pvalues.values]
        }
        summary_df = pd.DataFrame(summary_data)
        print(summary_df)
        print(f"\n調整済みR²: {model_sm.rsquared_adj:.4f}")
        print(f"F統計量: {model_sm.fvalue:.4f}, p値: {model_sm.f_pvalue:.6f}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].scatter(y_test, y_pred_test, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('実測値')
        axes[0, 0].set_ylabel('予測値')
        axes[0, 0].set_title(f'予測精度 (R²={test_r2:.3f})')
        
        residuals = y_test - y_pred_test
        axes[0, 1].scatter(y_pred_test, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('予測値')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差プロット')
        
        axes[1, 0].hist(residuals, bins=20, edgecolor='black')
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('頻度')
        axes[1, 0].set_title('残差の分布')
        
        top_vars = coef_df.head(10)
        axes[1, 1].barh(range(len(top_vars)), top_vars['回帰係数'].values)
        axes[1, 1].set_yticks(range(len(top_vars)))
        axes[1, 1].set_yticklabels(top_vars['変数'].values)
        axes[1, 1].set_xlabel('回帰係数')
        axes[1, 1].set_title('回帰係数（上位10変数）')
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('regression_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        self.results['single_regression'] = single_reg_df
        self.results['multiple_regression'] = {
            'coefficients': coef_df,
            'summary': summary_df,
            'metrics': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'adjusted_r2': model_sm.rsquared_adj,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
            }
        }
        
        return coef_df
    
    def pca_analysis(self):
        print("\n" + "="*50)
        print("主成分分析(PCA)")
        print("="*50)
        
        data = self.df[self.feature_cols].dropna()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(data)
        
        pca = PCA()
        pca.fit(X_scaled)
        
        explained_var = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(explained_var)
        
        n_components = np.argmax(cumsum_var >= 0.8) + 1
        print(f"\n累積寄与率80%を達成する主成分数: {n_components}")
        
        print("\n各主成分の寄与率:")
        for i, (var, cum) in enumerate(zip(explained_var[:5], cumsum_var[:5])):
            print(f"第{i+1}主成分: {var:.3f} (累積: {cum:.3f})")
        
        pca_reduced = PCA(n_components=min(3, len(self.feature_cols)))
        X_pca = pca_reduced.fit_transform(X_scaled)
        
        loadings = pd.DataFrame(
            pca_reduced.components_.T,
            index=self.feature_cols,
            columns=[f'PC{i+1}' for i in range(pca_reduced.n_components_)]
        )
        
        print("\n主成分負荷量（上位成分）:")
        print(loadings)
        
        for i in range(min(3, pca_reduced.n_components_)):
            print(f"\n第{i+1}主成分に強く寄与する変数:")
            important_vars = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5)
            for var, loading in important_vars.items():
                print(f"  {var}: {loadings.loc[var, f'PC{i+1}']:.3f}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].bar(range(1, min(11, len(explained_var)+1)), 
                      explained_var[:10], alpha=0.7, label='個別')
        axes[0, 0].plot(range(1, min(11, len(cumsum_var)+1)), 
                       cumsum_var[:10], 'ro-', label='累積')
        axes[0, 0].set_xlabel('主成分番号')
        axes[0, 0].set_ylabel('寄与率')
        axes[0, 0].set_title('主成分の寄与率')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        if pca_reduced.n_components_ >= 2:
            axes[0, 1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
            axes[0, 1].set_xlabel(f'第1主成分 ({explained_var[0]:.1%})')
            axes[0, 1].set_ylabel(f'第2主成分 ({explained_var[1]:.1%})')
            axes[0, 1].set_title('主成分得点の散布図')
            axes[0, 1].grid(True, alpha=0.3)
        
        sns.heatmap(loadings.T, cmap='coolwarm', center=0, 
                   annot=True, fmt='.2f', ax=axes[1, 0])
        axes[1, 0].set_title('主成分負荷量ヒートマップ')
        
        if self.target_col:
            y = self.df.loc[data.index, self.target_col]
            pca_df = pd.DataFrame(X_pca[:, :min(3, pca_reduced.n_components_)], 
                                columns=[f'PC{i+1}' for i in range(min(3, pca_reduced.n_components_))])
            pca_df[self.target_col] = y.values
            
            model = LinearRegression()
            X_pca_train = pca_df.drop(self.target_col, axis=1)
            y_pca_train = pca_df[self.target_col]
            model.fit(X_pca_train, y_pca_train)
            y_pred = model.predict(X_pca_train)
            r2 = r2_score(y_pca_train, y_pred)
            
            axes[1, 1].scatter(y_pca_train, y_pred, alpha=0.5)
            axes[1, 1].plot([y_pca_train.min(), y_pca_train.max()], 
                          [y_pca_train.min(), y_pca_train.max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('実測値')
            axes[1, 1].set_ylabel('予測値（主成分回帰）')
            axes[1, 1].set_title(f'主成分回帰 (R²={r2:.3f})')
            
            print(f"\n主成分回帰の結果:")
            print(f"使用主成分数: {X_pca_train.shape[1]}")
            print(f"R²スコア: {r2:.4f}")
        
        plt.tight_layout()
        plt.savefig('pca_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        self.results['pca'] = {
            'explained_variance': explained_var,
            'loadings': loadings,
            'n_components_80': n_components
        }
        
        return loadings
    
    def decision_tree_analysis(self):
        print("\n" + "="*50)
        print("決定木分析")
        print("="*50)
        
        data = self.df[[self.target_col] + self.feature_cols].dropna()
        X = data[self.feature_cols]
        y = data[self.target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        tree_model = DecisionTreeRegressor(max_depth=4, min_samples_split=20, 
                                          min_samples_leaf=10, random_state=42)
        tree_model.fit(X_train, y_train)
        
        y_pred_train = tree_model.predict(X_train)
        y_pred_test = tree_model.predict(X_test)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n決定木モデルの性能:")
        print(f"訓練データ R²: {train_r2:.4f}")
        print(f"テストデータ R²: {test_r2:.4f}")
        
        feature_importance = pd.DataFrame({
            '変数': self.feature_cols,
            '重要度': tree_model.feature_importances_
        }).sort_values('重要度', ascending=False)
        
        print("\n変数の重要度:")
        print(feature_importance)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        plot_tree(tree_model, feature_names=self.feature_cols, 
                 filled=True, rounded=True, fontsize=8, ax=axes[0])
        axes[0].set_title('決定木の構造')
        
        top_features = feature_importance.head(10)
        axes[1].barh(range(len(top_features)), top_features['重要度'].values)
        axes[1].set_yticks(range(len(top_features)))
        axes[1].set_yticklabels(top_features['変数'].values)
        axes[1].set_xlabel('重要度')
        axes[1].set_title('変数の重要度（上位10）')
        
        plt.tight_layout()
        plt.savefig('decision_tree_analysis.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        print("\n決定木による分岐ルール（主要なもの）:")
        tree_rules = self._extract_tree_rules(tree_model, self.feature_cols)
        for i, rule in enumerate(tree_rules[:5], 1):
            print(f"{i}. {rule}")
        
        self.results['decision_tree'] = {
            'feature_importance': feature_importance,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rules': tree_rules[:5]
        }
        
        return feature_importance
    
    def _extract_tree_rules(self, tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != -2 else "undefined!"
            for i in tree_.feature
        ]
        
        def recurse(node, depth, parent_rule=""):
            if tree_.feature[node] != -2:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                left_rule = f"{parent_rule} → {name} <= {threshold:.2f}"
                right_rule = f"{parent_rule} → {name} > {threshold:.2f}"
                
                left_rules = recurse(tree_.children_left[node], depth + 1, left_rule)
                right_rules = recurse(tree_.children_right[node], depth + 1, right_rule)
                
                return left_rules + right_rules
            else:
                value = tree_.value[node][0][0]
                return [f"{parent_rule} → 予測値: {value:.2f}"]
        
        rules = recurse(0, 1, "開始")
        return [r.replace("開始 → ", "") for r in rules if "予測値" in r][:10]
    
    def generate_report(self, output_file='analysis_report.txt'):
        print("\n" + "="*50)
        print("分析レポートの生成")
        print("="*50)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("製品シェア要因分析レポート\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"ターゲット変数: {self.target_col}\n")
            f.write(f"説明変数数: {len(self.feature_cols)}\n")
            f.write(f"サンプル数: {len(self.df)}\n\n")
            
            if 'correlations' in self.results:
                f.write("【相関分析の要約】\n")
                corr = self.results['correlations']
                significant = corr[corr['有意性(p<0.05)'] == 'あり']
                f.write(f"有意な相関を持つ変数数: {len(significant)}\n")
                if len(significant) > 0:
                    f.write("最も強い相関を持つ変数TOP3:\n")
                    for i, (idx, row) in enumerate(significant.head(3).iterrows(), 1):
                        f.write(f"  {i}. {idx}: r={row['Pearson相関係数']:.3f}\n")
                f.write("\n")
            
            if 'multiple_regression' in self.results:
                f.write("【回帰分析の要約】\n")
                metrics = self.results['multiple_regression']['metrics']
                f.write(f"調整済みR²: {metrics['adjusted_r2']:.4f}\n")
                f.write(f"テストデータR²: {metrics['test_r2']:.4f}\n")
                
                summary = self.results['multiple_regression']['summary']
                significant_vars = summary[summary['有意性'] != '']
                if len(significant_vars) > 1:
                    f.write(f"統計的に有意な変数数: {len(significant_vars)-1}\n")
                    f.write("有意な変数:\n")
                    for _, row in significant_vars.iterrows():
                        if row['変数'] != '定数項':
                            f.write(f"  - {row['変数']}: 係数={row['係数']:.3f} {row['有意性']}\n")
                f.write("\n")
            
            if 'pca' in self.results:
                f.write("【主成分分析の要約】\n")
                f.write(f"累積寄与率80%達成に必要な主成分数: {self.results['pca']['n_components_80']}\n")
                f.write("第1主成分の寄与率: {:.1%}\n".format(
                    self.results['pca']['explained_variance'][0]))
                f.write("\n")
            
            if 'decision_tree' in self.results:
                f.write("【決定木分析の要約】\n")
                f.write(f"テストデータR²: {self.results['decision_tree']['test_r2']:.4f}\n")
                f.write("重要度の高い変数TOP3:\n")
                for i, (_, row) in enumerate(
                    self.results['decision_tree']['feature_importance'].head(3).iterrows(), 1):
                    f.write(f"  {i}. {row['変数']}: {row['重要度']:.3f}\n")
                f.write("\n")
            
            f.write("【推奨アクション】\n")
            recommendations = self._generate_recommendations()
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"レポートを {output_file} に保存しました")
        return output_file
    
    def _generate_recommendations(self):
        recommendations = []
        
        if 'correlations' in self.results:
            corr = self.results['correlations']
            strong_corr = corr[abs(corr['Pearson相関係数']) > 0.5]
            if len(strong_corr) > 0:
                top_var = strong_corr.index[0]
                direction = "正" if strong_corr.iloc[0]['Pearson相関係数'] > 0 else "負"
                recommendations.append(
                    f"{top_var}と{direction}の強い相関があります。この変数の改善に注力することを推奨します。"
                )
        
        if 'multiple_regression' in self.results:
            summary = self.results['multiple_regression']['summary']
            sig_vars = summary[(summary['有意性'] != '') & (summary['変数'] != '定数項')]
            if len(sig_vars) > 0:
                top_impact = sig_vars.iloc[0]['変数']
                recommendations.append(
                    f"回帰分析により{top_impact}が統計的に有意な影響を持つことが確認されました。"
                )
        
        if 'decision_tree' in self.results:
            importance = self.results['decision_tree']['feature_importance']
            if importance.iloc[0]['重要度'] > 0.3:
                recommendations.append(
                    f"{importance.iloc[0]['変数']}が決定木分析で最も重要な変数として特定されました。"
                )
        
        if not recommendations:
            recommendations.append("さらに詳細な分析のため、追加データの収集を検討してください。")
        
        return recommendations
    
    def run_full_analysis(self):
        print("\n" + "="*50)
        print("完全分析の実行")
        print("="*50)
        
        self.basic_statistics()
        self.correlation_analysis()
        self.regression_analysis()
        self.pca_analysis()
        self.decision_tree_analysis()
        self.generate_report()
        
        print("\n" + "="*50)
        print("分析完了！")
        print("="*50)
        print("生成されたファイル:")
        print("  - correlation_analysis.png")
        print("  - regression_analysis.png")
        print("  - pca_analysis.png")
        print("  - decision_tree_analysis.png")
        print("  - analysis_report.txt")

def main():
    print("製品シェア要因分析ツール")
    print("="*50)
    
    csv_path = input("CSVファイルのパスを入力してください: ").strip()
    
    if not csv_path:
        print("サンプルデータを使用します")
        csv_path = "sample_data.csv"
    
    try:
        analyzer = ShareAnalyzer(csv_path)
        
        print("\n分析対象の設定")
        target = input("ターゲット変数（シェア列）の名前を入力してください: ").strip()
        
        use_all = input("すべての数値列を説明変数として使用しますか？ (y/n): ").strip().lower()
        
        if use_all == 'y':
            analyzer.set_target_and_features(target)
        else:
            print("説明変数をカンマ区切りで入力してください:")
            features = input().strip().split(',')
            features = [f.strip() for f in features]
            analyzer.set_target_and_features(target, features)
        
        analyzer.run_full_analysis()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()