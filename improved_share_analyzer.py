import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from scipy import stats
from scipy.stats import shapiro, levene, normaltest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings
warnings.filterwarnings('ignore')

class ImprovedShareAnalyzer:
    def __init__(self, csv_path):
        """改善版シェア分析クラス"""
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"データ読み込み完了: {len(self.df)}行 × {len(self.df.columns)}列")
        print(f"カラム: {', '.join(self.df.columns.tolist())}")
        self.target_col = None
        self.feature_cols = []
        self.results = {}
        self.df_processed = None
        
    def set_target_and_features(self, target_col, feature_cols=None):
        """ターゲット変数と説明変数を設定"""
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
        
        # 欠損値の状況を確認
        missing_cols = self.df[self.feature_cols + [self.target_col]].isnull().sum()
        if missing_cols.any():
            print("\n欠損値の状況:")
            print(missing_cols[missing_cols > 0])
    
    def detect_and_handle_outliers(self, method='iqr', threshold=1.5, handle='cap'):
        """外れ値の検出と処理"""
        print("\n" + "="*50)
        print("外れ値の検出と処理")
        print("="*50)
        
        self.df_processed = self.df.copy()
        outlier_info = {}
        
        for col in self.feature_cols + [self.target_col]:
            data = self.df_processed[col].dropna()
            
            if method == 'iqr':
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(data))
                outliers = z_scores > threshold
                lower_bound = data.mean() - threshold * data.std()
                upper_bound = data.mean() + threshold * data.std()
            
            else:
                raise ValueError("method must be 'iqr' or 'zscore'")
            
            n_outliers = outliers.sum()
            outlier_info[col] = {
                'n_outliers': n_outliers,
                'pct_outliers': n_outliers / len(data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            if n_outliers > 0:
                print(f"{col}: {n_outliers}個の外れ値を検出 ({n_outliers/len(data)*100:.1f}%)")
                
                if handle == 'cap':
                    # キャッピング（外れ値を境界値に置換）
                    self.df_processed.loc[self.df_processed[col] < lower_bound, col] = lower_bound
                    self.df_processed.loc[self.df_processed[col] > upper_bound, col] = upper_bound
                    print(f"  → キャッピング処理を実施")
                    
                elif handle == 'remove':
                    # 外れ値を含む行を削除
                    self.df_processed = self.df_processed[~outliers]
                    print(f"  → 外れ値を削除")
                    
                elif handle == 'none':
                    print(f"  → 処理なし")
        
        self.results['outlier_detection'] = outlier_info
        
        if handle != 'none':
            print(f"\n処理後のデータサイズ: {len(self.df_processed)}行")
        
        return outlier_info
    
    def check_multicollinearity(self):
        """多重共線性のチェック（VIF）"""
        print("\n" + "="*50)
        print("多重共線性チェック（VIF）")
        print("="*50)
        
        # データの準備
        df_work = self.df_processed if self.df_processed is not None else self.df
        data = df_work[self.feature_cols].dropna()
        
        # VIFの計算
        vif_data = pd.DataFrame()
        vif_data["Variable"] = self.feature_cols
        vif_values = []
        
        for i in range(len(self.feature_cols)):
            try:
                vif = variance_inflation_factor(data.values, i)
                vif_values.append(vif)
            except:
                vif_values.append(np.nan)
        
        vif_data["VIF"] = vif_values
        vif_data = vif_data.sort_values('VIF', ascending=False)
        
        print("\nVIF値（Variance Inflation Factor）:")
        print(vif_data)
        print("\n解釈:")
        print("VIF < 5: 多重共線性の問題なし")
        print("5 ≤ VIF < 10: 中程度の多重共線性")
        print("VIF ≥ 10: 深刻な多重共線性")
        
        # 問題のある変数を特定
        problematic_vars = vif_data[vif_data['VIF'] >= 10]
        if len(problematic_vars) > 0:
            print(f"\n⚠️ 深刻な多重共線性が検出された変数:")
            for _, row in problematic_vars.iterrows():
                print(f"  - {row['Variable']}: VIF = {row['VIF']:.2f}")
            print("\n推奨: これらの変数の削除または主成分分析の使用を検討してください")
        
        self.results['multicollinearity'] = vif_data
        return vif_data
    
    def check_statistical_assumptions(self):
        """統計的前提条件の検証"""
        print("\n" + "="*50)
        print("統計的前提条件の検証")
        print("="*50)
        
        df_work = self.df_processed if self.df_processed is not None else self.df
        data = df_work[[self.target_col] + self.feature_cols].dropna()
        
        assumptions_results = {}
        
        # 1. 正規性の検定
        print("\n1. 正規性検定（Shapiro-Wilk test）:")
        normality_results = {}
        for col in [self.target_col] + self.feature_cols[:5]:  # 最初の5変数のみ表示
            stat, p_value = shapiro(data[col]) if len(data[col]) < 5000 else normaltest(data[col])
            normality_results[col] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
            print(f"  {col}: p={p_value:.4f} {'✓ 正規分布' if p_value > 0.05 else '✗ 非正規分布'}")
        
        assumptions_results['normality'] = normality_results
        
        # 2. 線形性の確認（相関係数）
        print("\n2. 線形性の確認:")
        linearity_results = {}
        for col in self.feature_cols[:5]:  # 最初の5変数のみ表示
            corr, p_value = stats.pearsonr(data[col], data[self.target_col])
            linearity_results[col] = {
                'correlation': corr,
                'p_value': p_value
            }
            print(f"  {col}: r={corr:.3f}, p={p_value:.4f}")
        
        assumptions_results['linearity'] = linearity_results
        
        # 3. 等分散性の検定（回帰の残差で後で確認）
        print("\n3. 等分散性は回帰分析時に検証します")
        
        # 4. 独立性の確認（Durbin-Watson検定は回帰分析で実施）
        print("4. 独立性（自己相関）は回帰分析時に検証します")
        
        self.results['assumptions'] = assumptions_results
        return assumptions_results
    
    def regularized_regression_analysis(self, alpha_range=None):
        """正則化回帰分析（Ridge, Lasso, Elastic Net）"""
        print("\n" + "="*50)
        print("正則化回帰分析")
        print("="*50)
        
        df_work = self.df_processed if self.df_processed is not None else self.df
        data = df_work[[self.target_col] + self.feature_cols].dropna()
        X = data[self.feature_cols]
        y = data[self.target_col]
        
        # データの標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        if alpha_range is None:
            alpha_range = np.logspace(-4, 2, 20)
        
        results = {}
        
        # 1. Ridge回帰
        print("\n1. Ridge回帰:")
        ridge_cv = GridSearchCV(
            Ridge(), 
            {'alpha': alpha_range}, 
            cv=5, 
            scoring='r2'
        )
        ridge_cv.fit(X_train, y_train)
        best_ridge = ridge_cv.best_estimator_
        
        y_pred_ridge = best_ridge.predict(X_test)
        ridge_r2 = r2_score(y_test, y_pred_ridge)
        
        print(f"  最適α: {ridge_cv.best_params_['alpha']:.4f}")
        print(f"  テストR²: {ridge_r2:.4f}")
        
        results['ridge'] = {
            'best_alpha': ridge_cv.best_params_['alpha'],
            'test_r2': ridge_r2,
            'coefficients': dict(zip(self.feature_cols, best_ridge.coef_))
        }
        
        # 2. Lasso回帰
        print("\n2. Lasso回帰:")
        lasso_cv = GridSearchCV(
            Lasso(max_iter=1000), 
            {'alpha': alpha_range}, 
            cv=5, 
            scoring='r2'
        )
        lasso_cv.fit(X_train, y_train)
        best_lasso = lasso_cv.best_estimator_
        
        y_pred_lasso = best_lasso.predict(X_test)
        lasso_r2 = r2_score(y_test, y_pred_lasso)
        
        # ゼロでない係数の数
        n_selected = np.sum(best_lasso.coef_ != 0)
        
        print(f"  最適α: {lasso_cv.best_params_['alpha']:.4f}")
        print(f"  テストR²: {lasso_r2:.4f}")
        print(f"  選択された変数数: {n_selected}/{len(self.feature_cols)}")
        
        # 選択された変数を表示
        selected_vars = [var for var, coef in zip(self.feature_cols, best_lasso.coef_) if coef != 0]
        if selected_vars:
            print(f"  選択された変数: {', '.join(selected_vars[:5])}")
        
        results['lasso'] = {
            'best_alpha': lasso_cv.best_params_['alpha'],
            'test_r2': lasso_r2,
            'n_selected': n_selected,
            'selected_variables': selected_vars,
            'coefficients': dict(zip(self.feature_cols, best_lasso.coef_))
        }
        
        # 3. Elastic Net回帰
        print("\n3. Elastic Net回帰:")
        elastic_cv = GridSearchCV(
            ElasticNet(max_iter=1000), 
            {'alpha': alpha_range, 'l1_ratio': [0.1, 0.5, 0.7, 0.9]}, 
            cv=5, 
            scoring='r2'
        )
        elastic_cv.fit(X_train, y_train)
        best_elastic = elastic_cv.best_estimator_
        
        y_pred_elastic = best_elastic.predict(X_test)
        elastic_r2 = r2_score(y_test, y_pred_elastic)
        
        print(f"  最適α: {elastic_cv.best_params_['alpha']:.4f}")
        print(f"  最適l1_ratio: {elastic_cv.best_params_['l1_ratio']:.2f}")
        print(f"  テストR²: {elastic_r2:.4f}")
        
        results['elastic_net'] = {
            'best_alpha': elastic_cv.best_params_['alpha'],
            'best_l1_ratio': elastic_cv.best_params_['l1_ratio'],
            'test_r2': elastic_r2,
            'coefficients': dict(zip(self.feature_cols, best_elastic.coef_))
        }
        
        # 可視化
        self._plot_regularization_results(results, y_test, y_pred_ridge, y_pred_lasso, y_pred_elastic)
        
        self.results['regularized_regression'] = results
        return results
    
    def _plot_regularization_results(self, results, y_test, y_pred_ridge, y_pred_lasso, y_pred_elastic):
        """正則化回帰の結果を可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Ridge回帰の予測
        axes[0, 0].scatter(y_test, y_pred_ridge, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('実測値')
        axes[0, 0].set_ylabel('予測値')
        axes[0, 0].set_title(f'Ridge回帰 (R²={results["ridge"]["test_r2"]:.3f})')
        
        # Lasso回帰の予測
        axes[0, 1].scatter(y_test, y_pred_lasso, alpha=0.5)
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('実測値')
        axes[0, 1].set_ylabel('予測値')
        axes[0, 1].set_title(f'Lasso回帰 (R²={results["lasso"]["test_r2"]:.3f})')
        
        # Elastic Net回帰の予測
        axes[0, 2].scatter(y_test, y_pred_elastic, alpha=0.5)
        axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 2].set_xlabel('実測値')
        axes[0, 2].set_ylabel('予測値')
        axes[0, 2].set_title(f'Elastic Net (R²={results["elastic_net"]["test_r2"]:.3f})')
        
        # 係数の比較（上位10変数）
        ridge_coef = results['ridge']['coefficients']
        lasso_coef = results['lasso']['coefficients']
        elastic_coef = results['elastic_net']['coefficients']
        
        # 係数の絶対値で上位10変数を選択
        top_vars = sorted(ridge_coef.keys(), 
                         key=lambda x: abs(ridge_coef[x]), 
                         reverse=True)[:10]
        
        x_pos = np.arange(len(top_vars))
        width = 0.25
        
        axes[1, 0].bar(x_pos - width, [ridge_coef[v] for v in top_vars], 
                      width, label='Ridge', alpha=0.7)
        axes[1, 0].bar(x_pos, [lasso_coef[v] for v in top_vars], 
                      width, label='Lasso', alpha=0.7)
        axes[1, 0].bar(x_pos + width, [elastic_coef[v] for v in top_vars], 
                      width, label='Elastic', alpha=0.7)
        axes[1, 0].set_xlabel('変数')
        axes[1, 0].set_ylabel('係数')
        axes[1, 0].set_title('回帰係数の比較（上位10変数）')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(top_vars, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # R²スコアの比較
        models = ['Ridge', 'Lasso', 'Elastic Net']
        r2_scores = [results['ridge']['test_r2'], 
                    results['lasso']['test_r2'], 
                    results['elastic_net']['test_r2']]
        
        axes[1, 1].bar(models, r2_scores, color=['blue', 'green', 'orange'])
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_title('モデル性能の比較')
        axes[1, 1].set_ylim([min(r2_scores) * 0.9, min(1, max(r2_scores) * 1.1)])
        
        for i, v in enumerate(r2_scores):
            axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Lassoで選択された変数の可視化
        if results['lasso']['selected_variables']:
            selected = results['lasso']['selected_variables'][:10]
            selected_coef = [results['lasso']['coefficients'][v] for v in selected]
            
            axes[1, 2].barh(range(len(selected)), selected_coef)
            axes[1, 2].set_yticks(range(len(selected)))
            axes[1, 2].set_yticklabels(selected)
            axes[1, 2].set_xlabel('係数')
            axes[1, 2].set_title(f'Lassoで選択された変数 ({results["lasso"]["n_selected"]}個)')
            axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        plt.savefig('regularized_regression.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def cross_validation_analysis(self, cv_folds=5):
        """交差検証による性能評価"""
        print("\n" + "="*50)
        print(f"{cv_folds}分割交差検証")
        print("="*50)
        
        df_work = self.df_processed if self.df_processed is not None else self.df
        data = df_work[[self.target_col] + self.feature_cols].dropna()
        X = data[self.feature_cols]
        y = data[self.target_col]
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 交差検証の設定
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (α=1.0)': Ridge(alpha=1.0),
            'Lasso (α=0.1)': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        cv_results = {}
        
        for name, model in models.items():
            # R²スコア
            r2_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
            # RMSEスコア
            rmse_scores = -cross_val_score(model, X_scaled, y, cv=kfold, 
                                          scoring='neg_root_mean_squared_error')
            # MAEスコア
            mae_scores = -cross_val_score(model, X_scaled, y, cv=kfold, 
                                         scoring='neg_mean_absolute_error')
            
            cv_results[name] = {
                'r2_mean': r2_scores.mean(),
                'r2_std': r2_scores.std(),
                'rmse_mean': rmse_scores.mean(),
                'rmse_std': rmse_scores.std(),
                'mae_mean': mae_scores.mean(),
                'mae_std': mae_scores.std(),
                'r2_scores': r2_scores,
                'rmse_scores': rmse_scores,
                'mae_scores': mae_scores
            }
            
            print(f"\n{name}:")
            print(f"  R²: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
            print(f"  RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
            print(f"  MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std() * 2:.4f})")
        
        # 可視化
        self._plot_cv_results(cv_results)
        
        self.results['cross_validation'] = cv_results
        return cv_results
    
    def _plot_cv_results(self, cv_results):
        """交差検証結果の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        models = list(cv_results.keys())
        
        # R²スコアの箱ひげ図
        r2_data = [cv_results[m]['r2_scores'] for m in models]
        axes[0, 0].boxplot(r2_data, labels=models)
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('R²スコアの分布')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSEの箱ひげ図
        rmse_data = [cv_results[m]['rmse_scores'] for m in models]
        axes[0, 1].boxplot(rmse_data, labels=models)
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('RMSEの分布')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 平均スコアの比較（棒グラフ）
        x_pos = np.arange(len(models))
        width = 0.35
        
        r2_means = [cv_results[m]['r2_mean'] for m in models]
        r2_stds = [cv_results[m]['r2_std'] for m in models]
        
        axes[1, 0].bar(x_pos, r2_means, width, yerr=r2_stds, capsize=5, alpha=0.7)
        axes[1, 0].set_xlabel('モデル')
        axes[1, 0].set_ylabel('平均R²スコア')
        axes[1, 0].set_title('モデル性能の比較（平均±標準偏差）')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # スコアのばらつき（変動係数）
        cv_coefficients = [cv_results[m]['r2_std'] / cv_results[m]['r2_mean'] 
                          if cv_results[m]['r2_mean'] != 0 else 0 
                          for m in models]
        
        axes[1, 1].bar(x_pos, cv_coefficients, alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('モデル')
        axes[1, 1].set_ylabel('変動係数（CV）')
        axes[1, 1].set_title('予測の安定性（低いほど安定）')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cross_validation_results.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def advanced_regression_diagnostics(self):
        """回帰診断の詳細分析"""
        print("\n" + "="*50)
        print("回帰診断")
        print("="*50)
        
        df_work = self.df_processed if self.df_processed is not None else self.df
        data = df_work[[self.target_col] + self.feature_cols].dropna()
        X = data[self.feature_cols]
        y = data[self.target_col]
        
        # statsmodelsで回帰モデルを構築
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        
        print("\n回帰モデルサマリー:")
        print(f"調整済みR²: {model.rsquared_adj:.4f}")
        print(f"F統計量: {model.fvalue:.4f}, p値: {model.f_pvalue:.6f}")
        print(f"AIC: {model.aic:.2f}, BIC: {model.bic:.2f}")
        
        # 1. 残差の正規性検定
        residuals = model.resid
        stat, p_value = shapiro(residuals) if len(residuals) < 5000 else normaltest(residuals)
        print(f"\n残差の正規性検定: p値={p_value:.4f}")
        print(f"  {'✓ 残差は正規分布に従う' if p_value > 0.05 else '✗ 残差は正規分布に従わない'}")
        
        # 2. 等分散性の検定（Breusch-Pagan検定）
        bp_test = het_breuschpagan(residuals, X_sm)
        print(f"\nBreusch-Pagan検定（等分散性）: p値={bp_test[1]:.4f}")
        print(f"  {'✓ 等分散性の仮定を満たす' if bp_test[1] > 0.05 else '✗ 不等分散の可能性'}")
        
        # 3. Durbin-Watson検定（自己相関）
        dw_stat = sm.stats.durbin_watson(residuals)
        print(f"\nDurbin-Watson統計量: {dw_stat:.4f}")
        print(f"  解釈: ", end="")
        if dw_stat < 1.5:
            print("正の自己相関の可能性")
        elif dw_stat > 2.5:
            print("負の自己相関の可能性")
        else:
            print("自己相関なし（良好）")
        
        # 4. 影響力のある観測値の検出
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        n_influential = np.sum(cooks_d > 4/len(data))
        
        print(f"\nCook's distance（影響力のある観測値）:")
        print(f"  閾値(4/n)を超える観測値: {n_influential}個")
        
        # 可視化
        self._plot_regression_diagnostics(model, residuals, cooks_d)
        
        diagnostics_results = {
            'adj_r2': model.rsquared_adj,
            'aic': model.aic,
            'bic': model.bic,
            'normality_p': p_value,
            'homoscedasticity_p': bp_test[1],
            'durbin_watson': dw_stat,
            'n_influential': n_influential
        }
        
        self.results['regression_diagnostics'] = diagnostics_results
        return diagnostics_results
    
    def _plot_regression_diagnostics(self, model, residuals, cooks_d):
        """回帰診断の可視化"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        fitted_values = model.fittedvalues
        
        # 1. 残差プロット
        axes[0, 0].scatter(fitted_values, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('予測値')
        axes[0, 0].set_ylabel('残差')
        axes[0, 0].set_title('残差プロット')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Qプロット
        sm.qqplot(residuals, line='45', ax=axes[0, 1])
        axes[0, 1].set_title('Q-Qプロット（正規性の確認）')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. スケール-位置プロット
        standardized_residuals = residuals / residuals.std()
        axes[0, 2].scatter(fitted_values, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
        axes[0, 2].set_xlabel('予測値')
        axes[0, 2].set_ylabel('√|標準化残差|')
        axes[0, 2].set_title('スケール-位置プロット（等分散性）')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 残差のヒストグラム
        axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('残差')
        axes[1, 0].set_ylabel('頻度')
        axes[1, 0].set_title('残差の分布')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Cook's distance
        n = len(cooks_d)
        axes[1, 1].stem(range(n), cooks_d, markerfmt=',', basefmt=' ')
        axes[1, 1].axhline(y=4/n, color='r', linestyle='--', label=f'閾値 (4/n={4/n:.3f})')
        axes[1, 1].set_xlabel('観測値インデックス')
        axes[1, 1].set_ylabel("Cook's Distance")
        axes[1, 1].set_title("Cook's Distance（影響力のある観測値）")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. レバレッジ vs 標準化残差
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag
        axes[1, 2].scatter(leverage, standardized_residuals, alpha=0.5)
        axes[1, 2].axhline(y=0, color='r', linestyle='--')
        axes[1, 2].set_xlabel('レバレッジ')
        axes[1, 2].set_ylabel('標準化残差')
        axes[1, 2].set_title('レバレッジ vs 残差')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_diagnostics.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def generate_improved_report(self, output_file='improved_analysis_report.txt'):
        """改善版の分析レポート生成"""
        print("\n" + "="*50)
        print("分析レポートの生成")
        print("="*50)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("改善版 製品シェア要因分析レポート\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"ターゲット変数: {self.target_col}\n")
            f.write(f"説明変数数: {len(self.feature_cols)}\n")
            f.write(f"サンプル数: {len(self.df)}\n\n")
            
            # 外れ値検出結果
            if 'outlier_detection' in self.results:
                f.write("【外れ値検出】\n")
                outliers = self.results['outlier_detection']
                total_outliers = sum(info['n_outliers'] for info in outliers.values())
                f.write(f"検出された外れ値総数: {total_outliers}\n\n")
            
            # 多重共線性チェック
            if 'multicollinearity' in self.results:
                f.write("【多重共線性チェック】\n")
                vif_data = self.results['multicollinearity']
                problematic = vif_data[vif_data['VIF'] >= 10]
                if len(problematic) > 0:
                    f.write("⚠️ VIF≥10の変数:\n")
                    for _, row in problematic.iterrows():
                        f.write(f"  - {row['Variable']}: VIF={row['VIF']:.2f}\n")
                else:
                    f.write("✓ 深刻な多重共線性は検出されませんでした\n")
                f.write("\n")
            
            # 統計的前提条件
            if 'assumptions' in self.results:
                f.write("【統計的前提条件の検証】\n")
                assumptions = self.results['assumptions']
                
                # 正規性
                if 'normality' in assumptions:
                    normal_vars = sum(1 for v in assumptions['normality'].values() if v['is_normal'])
                    total_vars = len(assumptions['normality'])
                    f.write(f"正規性を満たす変数: {normal_vars}/{total_vars}\n")
                f.write("\n")
            
            # 交差検証結果
            if 'cross_validation' in self.results:
                f.write("【交差検証による性能評価】\n")
                cv_results = self.results['cross_validation']
                
                # 最良モデルを特定
                best_model = max(cv_results.keys(), key=lambda x: cv_results[x]['r2_mean'])
                best_score = cv_results[best_model]['r2_mean']
                best_std = cv_results[best_model]['r2_std']
                
                f.write(f"最良モデル: {best_model}\n")
                f.write(f"  R²スコア: {best_score:.4f} (±{best_std*2:.4f})\n\n")
                
                f.write("全モデルの性能:\n")
                for model_name, scores in cv_results.items():
                    f.write(f"  {model_name}: R²={scores['r2_mean']:.4f}, RMSE={scores['rmse_mean']:.4f}\n")
                f.write("\n")
            
            # 正則化回帰結果
            if 'regularized_regression' in self.results:
                f.write("【正則化回帰分析】\n")
                reg_results = self.results['regularized_regression']
                
                f.write("モデル性能:\n")
                f.write(f"  Ridge R²: {reg_results['ridge']['test_r2']:.4f}\n")
                f.write(f"  Lasso R²: {reg_results['lasso']['test_r2']:.4f}\n")
                f.write(f"  Elastic Net R²: {reg_results['elastic_net']['test_r2']:.4f}\n")
                
                if 'selected_variables' in reg_results['lasso']:
                    f.write(f"\nLassoで選択された重要変数（{reg_results['lasso']['n_selected']}個）:\n")
                    for var in reg_results['lasso']['selected_variables'][:5]:
                        f.write(f"  - {var}\n")
                f.write("\n")
            
            # 回帰診断
            if 'regression_diagnostics' in self.results:
                f.write("【回帰診断】\n")
                diag = self.results['regression_diagnostics']
                
                f.write(f"調整済みR²: {diag['adj_r2']:.4f}\n")
                f.write(f"AIC: {diag['aic']:.2f}, BIC: {diag['bic']:.2f}\n")
                
                f.write("\n前提条件の検証:\n")
                f.write(f"  残差の正規性: {'✓' if diag['normality_p'] > 0.05 else '✗'} (p={diag['normality_p']:.4f})\n")
                f.write(f"  等分散性: {'✓' if diag['homoscedasticity_p'] > 0.05 else '✗'} (p={diag['homoscedasticity_p']:.4f})\n")
                f.write(f"  自己相関: DW統計量={diag['durbin_watson']:.4f}\n")
                f.write(f"  影響力のある観測値: {diag['n_influential']}個\n")
                f.write("\n")
            
            # 推奨事項
            f.write("【推奨事項】\n")
            recommendations = self._generate_improved_recommendations()
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"レポートを {output_file} に保存しました")
        return output_file
    
    def _generate_improved_recommendations(self):
        """改善版の推奨事項生成"""
        recommendations = []
        
        # 多重共線性の問題がある場合
        if 'multicollinearity' in self.results:
            vif_data = self.results['multicollinearity']
            problematic = vif_data[vif_data['VIF'] >= 10]
            if len(problematic) > 0:
                recommendations.append(
                    f"多重共線性の問題が検出されました。{problematic.iloc[0]['Variable']}などの変数の削除または主成分分析の使用を検討してください。"
                )
        
        # 交差検証の結果に基づく推奨
        if 'cross_validation' in self.results:
            cv_results = self.results['cross_validation']
            best_model = max(cv_results.keys(), key=lambda x: cv_results[x]['r2_mean'])
            
            if 'Random Forest' in best_model or 'Gradient' in best_model:
                recommendations.append(
                    f"非線形モデル（{best_model}）が最良の性能を示しています。変数間の非線形関係を考慮した分析が有効です。"
                )
            elif 'Lasso' in best_model:
                recommendations.append(
                    "Lasso回帰が最良の性能を示しています。変数選択による解釈しやすいモデルの構築が可能です。"
                )
        
        # 正則化回帰の結果に基づく推奨
        if 'regularized_regression' in self.results:
            reg_results = self.results['regularized_regression']
            if reg_results['lasso']['n_selected'] < len(self.feature_cols) * 0.5:
                recommendations.append(
                    f"Lasso回帰により{reg_results['lasso']['n_selected']}個の重要変数が特定されました。これらの変数に焦点を当てた施策を検討してください。"
                )
        
        # 回帰診断の結果に基づく推奨
        if 'regression_diagnostics' in self.results:
            diag = self.results['regression_diagnostics']
            if diag['normality_p'] < 0.05:
                recommendations.append(
                    "残差が正規分布に従わないため、変数変換（対数変換など）またはロバスト回帰の使用を検討してください。"
                )
            if diag['homoscedasticity_p'] < 0.05:
                recommendations.append(
                    "不等分散が検出されました。加重最小二乗法または変数変換を検討してください。"
                )
            if diag['n_influential'] > 0:
                recommendations.append(
                    f"{diag['n_influential']}個の影響力の大きい観測値が検出されました。これらのデータポイントを詳細に確認してください。"
                )
        
        if not recommendations:
            recommendations.append("分析結果は概ね良好です。さらなる精度向上のため、追加の説明変数の収集を検討してください。")
        
        return recommendations
    
    def run_improved_analysis(self):
        """改善版の完全分析実行"""
        print("\n" + "="*50)
        print("改善版 完全分析の実行")
        print("="*50)
        
        # 1. 外れ値の検出と処理
        self.detect_and_handle_outliers(method='iqr', threshold=1.5, handle='cap')
        
        # 2. 多重共線性のチェック
        self.check_multicollinearity()
        
        # 3. 統計的前提条件の検証
        self.check_statistical_assumptions()
        
        # 4. 交差検証
        self.cross_validation_analysis(cv_folds=5)
        
        # 5. 正則化回帰
        self.regularized_regression_analysis()
        
        # 6. 回帰診断
        self.advanced_regression_diagnostics()
        
        # 7. レポート生成
        self.generate_improved_report()
        
        print("\n" + "="*50)
        print("改善版分析完了！")
        print("="*50)
        print("生成されたファイル:")
        print("  - regularized_regression.png")
        print("  - cross_validation_results.png")
        print("  - regression_diagnostics.png")
        print("  - improved_analysis_report.txt")

def main():
    print("改善版 製品シェア要因分析ツール")
    print("="*50)
    
    csv_path = input("CSVファイルのパスを入力してください: ").strip()
    
    if not csv_path:
        print("デフォルトファイルを使用します")
        csv_path = "ダミーデータ（CSV）.csv"
    
    try:
        analyzer = ImprovedShareAnalyzer(csv_path)
        
        print("\n分析対象の設定")
        target = input("ターゲット変数（シェア列）の名前を入力してください [製品Aシェア]: ").strip()
        if not target:
            target = "製品Aシェア"
        
        use_all = input("すべての数値列を説明変数として使用しますか？ (y/n) [y]: ").strip().lower()
        if not use_all:
            use_all = 'y'
        
        if use_all == 'y':
            analyzer.set_target_and_features(target)
        else:
            print("説明変数をカンマ区切りで入力してください:")
            features = input().strip().split(',')
            features = [f.strip() for f in features]
            analyzer.set_target_and_features(target, features)
        
        analyzer.run_improved_analysis()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()