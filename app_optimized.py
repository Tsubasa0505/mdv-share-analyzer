"""
MDV Share Analyzer - Optimized Version
製品シェア要因分析ツール（最適化版）
Performance optimized with caching, modular structure, and better error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import io
import json
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from functools import lru_cache
import hashlib

warnings.filterwarnings('ignore')

# ========================
# Configuration Constants
# ========================
class Config:
    """Application configuration constants"""
    PAGE_TITLE = "製品シェア要因分析ツール"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Display parameters
    MAX_PREVIEW_ROWS = 10
    CHART_HEIGHT = 400
    CHART_WIDTH = 800
    
    # Cache settings
    CACHE_TTL = 3600  # 1 hour
    
    # Encoding options
    ENCODINGS = ["utf-8", "shift-jis", "cp932", "utf-8-sig"]
    DEFAULT_ENCODING_INDEX = 1

# ========================
# Data Classes
# ========================
@dataclass
class AnalysisResults:
    """Container for analysis results"""
    correlation_matrix: pd.DataFrame
    regression_results: Dict[str, Any]
    tree_results: Dict[str, Any]
    pca_results: Dict[str, Any]
    vif_results: Optional[pd.DataFrame] = None
    outliers: Optional[Dict[str, Any]] = None
    cross_validation_scores: Optional[Dict[str, float]] = None

# ========================
# Utility Functions
# ========================
@st.cache_data(ttl=Config.CACHE_TTL)
def load_and_preprocess_data(file_content: bytes, encoding: str) -> pd.DataFrame:
    """Load and preprocess CSV data with caching"""
    try:
        df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
        
        # Convert percentage strings to numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].astype(str).str.contains('%').any():
                    df[col] = (df[col].str.replace('%', '')
                              .replace('-', '0')
                              .astype(float))
        
        # Remove columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        # Fill NaN values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    except Exception as e:
        st.error(f"データ読み込みエラー: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Calculate Variance Inflation Factor for multicollinearity detection"""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_values = []
    
    for i in range(len(X.columns)):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(vif if vif < 1000 else np.inf)
        except:
            vif_values.append(np.nan)
    
    vif_data["VIF"] = vif_values
    vif_data["Multicollinearity"] = vif_data["VIF"].apply(
        lambda x: "High" if x > 10 else ("Moderate" if x > 5 else "Low")
    )
    return vif_data

@st.cache_data
def detect_outliers(data: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """Detect outliers using IQR and Z-score methods"""
    outliers = {}
    
    for col in columns:
        # IQR method
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data[col].dropna()))
        
        outliers[col] = {
            'iqr_outliers': data[(data[col] < lower_bound) | (data[col] > upper_bound)].index.tolist(),
            'z_outliers': data.index[z_scores > 3].tolist() if len(z_scores) > 0 else [],
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers

# ========================
# Analysis Functions
# ========================
@st.cache_data
def perform_correlation_analysis(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Perform correlation analysis with caching"""
    return df[columns].corr()

@st.cache_data
def perform_regression_analysis(X: pd.DataFrame, y: pd.Series, 
                               model_type: str = "linear") -> Dict[str, Any]:
    """Perform regression analysis with multiple models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # Select model
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1.0),
        "elastic": ElasticNet(alpha=1.0)
    }
    
    model = models.get(model_type, LinearRegression())
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=Config.CV_FOLDS, 
                               scoring='r2')
    
    # Statsmodels for detailed statistics
    X_sm = sm.add_constant(X_train)
    model_sm = sm.OLS(y_train, X_sm).fit()
    
    # Feature importance
    if hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'Variable': X.columns,
            'Coefficient': model.coef_,
            'Abs_Coefficient': np.abs(model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
    else:
        feature_importance = None
    
    return {
        'model': model,
        'model_sm': model_sm,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_test': y_test,
        'y_pred': y_pred_test,
        'feature_importance': feature_importance,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train
    }

@st.cache_data
def perform_tree_analysis(X: pd.DataFrame, y: pd.Series, 
                         use_random_forest: bool = False) -> Dict[str, Any]:
    """Perform decision tree or random forest analysis"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    # Select model
    if use_random_forest:
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=Config.RANDOM_STATE
        )
    else:
        model = DecisionTreeRegressor(
            max_depth=5,
            random_state=Config.RANDOM_STATE
        )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Feature importance
    importance_df = pd.DataFrame({
        'Variable': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'y_test': y_test,
        'y_pred': y_pred,
        'feature_importance': importance_df
    }

@st.cache_data
def perform_pca_analysis(X: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
    """Perform PCA analysis"""
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(n_components, X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    
    # Create component dataframe
    components_df = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
    )
    
    # Feature loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=X.columns
    )
    
    return {
        'pca': pca,
        'scaler': scaler,
        'X_pca': X_pca,
        'components_df': components_df,
        'loadings': loadings,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_)
    }

# ========================
# Visualization Functions
# ========================
def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create an interactive correlation heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="相関係数")
    ))
    
    fig.update_layout(
        title="相関行列ヒートマップ",
        height=Config.CHART_HEIGHT,
        xaxis_title="",
        yaxis_title=""
    )
    
    return fig

def create_regression_plots(results: Dict[str, Any]) -> go.Figure:
    """Create regression analysis plots"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("予測値 vs 実測値", "残差プロット")
    )
    
    # Prediction vs Actual
    fig.add_trace(
        go.Scatter(
            x=results['y_test'],
            y=results['y_pred'],
            mode='markers',
            name='予測値',
            marker=dict(size=8, opacity=0.6)
        ),
        row=1, col=1
    )
    
    # Perfect prediction line
    min_val = min(results['y_test'].min(), results['y_pred'].min())
    max_val = max(results['y_test'].max(), results['y_pred'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='理想線',
            line=dict(dash='dash', color='red')
        ),
        row=1, col=1
    )
    
    # Residuals
    residuals = results['y_test'] - results['y_pred']
    fig.add_trace(
        go.Scatter(
            x=results['y_pred'],
            y=residuals,
            mode='markers',
            name='残差',
            marker=dict(size=8, opacity=0.6)
        ),
        row=1, col=2
    )
    
    # Zero line for residuals
    fig.add_trace(
        go.Scatter(
            x=[results['y_pred'].min(), results['y_pred'].max()],
            y=[0, 0],
            mode='lines',
            name='ゼロライン',
            line=dict(dash='dash', color='red')
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="実測値", row=1, col=1)
    fig.update_yaxes(title_text="予測値", row=1, col=1)
    fig.update_xaxes(title_text="予測値", row=1, col=2)
    fig.update_yaxes(title_text="残差", row=1, col=2)
    
    fig.update_layout(
        height=Config.CHART_HEIGHT,
        showlegend=True,
        title_text=f"回帰分析結果 (R²={results['test_r2']:.4f})"
    )
    
    return fig

def create_feature_importance_chart(importance_df: pd.DataFrame, 
                                   chart_type: str = "bar") -> go.Figure:
    """Create feature importance visualization"""
    importance_df = importance_df.head(10)  # Top 10 features
    
    if chart_type == "bar":
        fig = px.bar(
            importance_df,
            x='Importance' if 'Importance' in importance_df.columns else 'Abs_Coefficient',
            y='Variable',
            orientation='h',
            title="特徴量重要度（上位10変数）"
        )
    else:
        fig = go.Figure(go.Pie(
            labels=importance_df['Variable'],
            values=importance_df['Importance'] if 'Importance' in importance_df.columns else importance_df['Abs_Coefficient'],
            hole=0.3
        ))
        fig.update_layout(title="特徴量重要度の割合")
    
    fig.update_layout(height=Config.CHART_HEIGHT)
    return fig

def create_pca_plots(pca_results: Dict[str, Any]) -> Tuple[go.Figure, go.Figure]:
    """Create PCA visualization plots"""
    # Scree plot
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(len(pca_results['explained_variance_ratio']))],
        y=pca_results['explained_variance_ratio'],
        name='寄与率'
    ))
    fig_scree.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(len(pca_results['cumulative_variance_ratio']))],
        y=pca_results['cumulative_variance_ratio'],
        mode='lines+markers',
        name='累積寄与率',
        yaxis='y2'
    ))
    
    fig_scree.update_layout(
        title="主成分の寄与率",
        yaxis=dict(title='寄与率'),
        yaxis2=dict(title='累積寄与率', overlaying='y', side='right'),
        height=Config.CHART_HEIGHT
    )
    
    # Biplot
    if pca_results['X_pca'].shape[1] >= 2:
        fig_biplot = go.Figure()
        
        # Scatter plot of samples
        fig_biplot.add_trace(go.Scatter(
            x=pca_results['X_pca'][:, 0],
            y=pca_results['X_pca'][:, 1],
            mode='markers',
            name='サンプル',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Loading vectors
        loadings = pca_results['loadings']
        for idx, var in enumerate(loadings.index):
            fig_biplot.add_annotation(
                x=loadings.iloc[idx, 0] * 3,
                y=loadings.iloc[idx, 1] * 3,
                ax=0, ay=0,
                xref="x", yref="y",
                axref="x", ayref="y",
                text=var,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red"
            )
        
        fig_biplot.update_layout(
            title="PCA Biplot",
            xaxis_title=f"PC1 ({pca_results['explained_variance_ratio'][0]:.1%})",
            yaxis_title=f"PC2 ({pca_results['explained_variance_ratio'][1]:.1%})",
            height=Config.CHART_HEIGHT
        )
    else:
        fig_biplot = go.Figure()
        fig_biplot.add_annotation(
            text="2つ以上の主成分が必要です",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
    
    return fig_scree, fig_biplot

# ========================
# What-If Simulation
# ========================
def perform_what_if_simulation(model: Any, feature_names: List[str], 
                              base_values: Dict[str, float],
                              adjustments: Dict[str, float]) -> Dict[str, Any]:
    """Perform What-If simulation with sensitivity analysis"""
    # Create adjusted input
    adjusted_values = base_values.copy()
    for feature, adjustment in adjustments.items():
        if feature in adjusted_values:
            adjusted_values[feature] = adjustment
    
    # Convert to array for prediction
    X_base = np.array([base_values[f] for f in feature_names]).reshape(1, -1)
    X_adjusted = np.array([adjusted_values[f] for f in feature_names]).reshape(1, -1)
    
    # Predictions
    base_prediction = model.predict(X_base)[0]
    adjusted_prediction = model.predict(X_adjusted)[0]
    
    # Sensitivity analysis
    sensitivity = {}
    for feature in feature_names:
        X_temp = X_base.copy()
        X_temp[0, feature_names.index(feature)] *= 1.1  # 10% increase
        temp_prediction = model.predict(X_temp)[0]
        sensitivity[feature] = (temp_prediction - base_prediction) / base_prediction * 100
    
    return {
        'base_prediction': base_prediction,
        'adjusted_prediction': adjusted_prediction,
        'change': adjusted_prediction - base_prediction,
        'change_percent': (adjusted_prediction - base_prediction) / base_prediction * 100,
        'sensitivity': sensitivity,
        'adjusted_values': adjusted_values
    }

# ========================
# AI Prompt Generation
# ========================
def generate_analysis_prompt(analysis_results: Dict[str, Any], 
                           context: str = "",
                           level: str = "一般") -> str:
    """Generate AI prompt for analysis interpretation"""
    # Format analysis results
    results_json = json.dumps(analysis_results, ensure_ascii=False, indent=2, default=str)
    
    # Level-specific instructions
    level_instructions = {
        "初心者": "専門用語を使わず、誰にでも分かるように説明してください。",
        "一般": "一般的なビジネスパーソンが理解できるレベルで説明してください。",
        "専門家": "統計的な詳細も含めて、専門的に解説してください。"
    }
    
    prompt = f"""
以下のデータ分析結果を解釈して、分かりやすく説明してください。

{level_instructions.get(level, level_instructions["一般"])}

## 分析の背景
{context if context else "製品やサービスのシェアに影響する要因を分析しています。"}

## 分析結果
{results_json}

## 解釈してほしいポイント

1. **この分析結果から何が分かるか**
   - 最も重要な発見は何か
   - どの要因が最も影響力があるか
   
2. **なぜその結果になったのか**
   - 統計的に見て信頼できる結果か
   - 変数間の関係性をどう理解すべきか
   
3. **実務への活用方法**
   - この結果をどう活用すればよいか
   - 優先的に取り組むべきことは何か
   
4. **注意すべき点**
   - この分析の限界は何か
   - 誤解しやすい点はあるか

分析結果を見て、上記のポイントについて解説をお願いします。
"""
    
    return prompt

# ========================
# Statistics Guide
# ========================
def show_statistics_guide():
    """Display statistics guide"""
    st.markdown("""
    <style>
    .stat-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stat-title {
        color: #1f77b4;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .formula {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📚 統計ガイド")
    
    with st.expander("📊 基本統計量", expanded=False):
        st.markdown("""
        ### 平均（Mean）
        データの中心傾向を表す最も基本的な指標
        
        ### 中央値（Median）
        データを大きさ順に並べたときの中央の値
        
        ### 標準偏差（Standard Deviation）
        データのばらつきの大きさを表す指標
        
        ### 四分位数（Quartiles）
        - Q1（第1四分位数）：下位25%の位置
        - Q2（第2四分位数）：中央値
        - Q3（第3四分位数）：上位25%の位置
        """)
    
    with st.expander("🔗 相関分析", expanded=False):
        st.markdown("""
        ### 相関係数（Correlation Coefficient）
        - **範囲**: -1 ～ 1
        - **解釈**:
          - 0.7以上: 強い正の相関
          - 0.4～0.7: 中程度の正の相関
          - -0.4～0.4: 弱い相関
          - -0.7～-0.4: 中程度の負の相関
          - -0.7以下: 強い負の相関
        
        ### 注意点
        - 相関関係は因果関係を意味しない
        - 外れ値の影響を受けやすい
        - 非線形な関係は検出できない
        """)
    
    with st.expander("📈 回帰分析", expanded=False):
        st.markdown("""
        ### R²スコア（決定係数）
        - モデルの説明力を表す指標（0～1）
        - 0.7以上: 良好なモデル
        - 0.5～0.7: まずまずのモデル
        - 0.5未満: 改善が必要
        
        ### RMSE（二乗平均平方根誤差）
        予測誤差の大きさを表す指標（小さいほど良い）
        
        ### p値（P-value）
        - 0.05未満: 統計的に有意
        - 0.05以上: 統計的に有意でない
        
        ### 回帰係数
        各説明変数が目的変数に与える影響の大きさ
        """)
    
    with st.expander("🌳 決定木分析", expanded=False):
        st.markdown("""
        ### 特徴量重要度（Feature Importance）
        各変数の予測への貢献度（0～1、合計1）
        
        ### 利点
        - 非線形な関係を捉えられる
        - 解釈しやすい
        - 外れ値に強い
        
        ### 欠点
        - 過学習しやすい
        - 不安定（データの小さな変化で結果が大きく変わる）
        """)
    
    with st.expander("🎯 主成分分析（PCA）", expanded=False):
        st.markdown("""
        ### 寄与率（Explained Variance Ratio）
        各主成分が説明する分散の割合
        
        ### 累積寄与率
        - 70%以上: 十分な情報を保持
        - 80%以上: 良好
        - 90%以上: 非常に良好
        
        ### 主成分負荷量（Loadings）
        各変数が主成分に与える影響の大きさ
        """)
    
    with st.expander("⚠️ 多重共線性（VIF）", expanded=False):
        st.markdown("""
        ### VIF（Variance Inflation Factor）
        - **1～5**: 多重共線性なし
        - **5～10**: 中程度の多重共線性
        - **10以上**: 深刻な多重共線性
        
        ### 対処法
        - 相関の高い変数の一方を除外
        - 主成分分析で次元削減
        - Ridge回帰やLasso回帰を使用
        """)

# ========================
# Main Application
# ========================
def main():
    """Main application entry point"""
    # Page configuration
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT
    )
    
    st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
    st.markdown("---")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    
    # Sidebar
    with st.sidebar:
        st.header("📁 データアップロード")
        
        uploaded_file = st.file_uploader(
            "CSVファイルを選択してください",
            type=['csv'],
            help="分析したいCSVファイルをアップロードしてください"
        )
        
        encoding = st.selectbox(
            "文字エンコーディング",
            Config.ENCODINGS,
            index=Config.DEFAULT_ENCODING_INDEX,
            help="日本語のCSVファイルの場合は'shift-jis'を選択してください"
        )
        
        st.markdown("---")
        
        # Analysis settings
        if uploaded_file is not None:
            st.header("⚙️ 分析設定")
            
            model_type = st.selectbox(
                "回帰モデル",
                ["linear", "ridge", "lasso", "elastic"],
                format_func=lambda x: {
                    "linear": "線形回帰",
                    "ridge": "Ridge回帰",
                    "lasso": "Lasso回帰",
                    "elastic": "Elastic Net"
                }[x]
            )
            
            use_random_forest = st.checkbox(
                "ランダムフォレストを使用",
                help="決定木の代わりにランダムフォレストを使用します"
            )
            
            perform_vif = st.checkbox(
                "VIF分析を実行",
                value=True,
                help="多重共線性をチェックします"
            )
            
            detect_outliers_flag = st.checkbox(
                "外れ値検出を実行",
                value=True,
                help="IQRとZ-scoreで外れ値を検出します"
            )
        
        st.markdown("---")
        st.info(
            "💡 **使い方**\n"
            "1. CSVファイルをアップロード\n"
            "2. ターゲット変数を選択\n"
            "3. 説明変数を選択\n"
            "4. 分析を実行"
        )
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load data with caching
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            df = load_and_preprocess_data(file_content, encoding)
            
            if df.empty:
                st.error("データの読み込みに失敗しました")
                return
            
            # Data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("行数", f"{len(df):,}")
            with col2:
                st.metric("列数", f"{len(df.columns):,}")
            with col3:
                st.metric("欠損値", f"{df.isnull().sum().sum():,}")
            with col4:
                st.metric("数値列", f"{len(df.select_dtypes(include=[np.number]).columns):,}")
            
            # Data preview
            with st.expander("📋 データプレビュー", expanded=False):
                st.dataframe(df.head(Config.MAX_PREVIEW_ROWS))
                
                st.subheader("基本統計量")
                st.dataframe(df.describe())
            
            st.markdown("---")
            
            # Variable selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    target_col = st.selectbox(
                        "🎯 ターゲット変数（目的変数）",
                        numeric_cols,
                        help="予測したい変数を選択してください"
                    )
                
                with col2:
                    feature_cols = st.multiselect(
                        "📊 説明変数",
                        [col for col in numeric_cols if col != target_col],
                        default=[col for col in numeric_cols if col != target_col][:5],
                        help="分析に使用する変数を選択してください"
                    )
                
                if len(feature_cols) > 0:
                    # Analysis button
                    if st.button("🚀 分析を実行", type="primary", use_container_width=True):
                        with st.spinner("分析中..."):
                            # Prepare data
                            X = df[feature_cols]
                            y = df[target_col]
                            
                            # Remove rows with NaN
                            mask = ~(X.isnull().any(axis=1) | y.isnull())
                            X = X[mask]
                            y = y[mask]
                            
                            # Initialize results container
                            analysis_summary = {
                                "データ概要": {
                                    "サンプル数": len(X),
                                    "説明変数数": len(feature_cols),
                                    "ターゲット変数": target_col
                                }
                            }
                            
                            # Correlation analysis
                            corr_matrix = perform_correlation_analysis(df, feature_cols + [target_col])
                            
                            # Regression analysis
                            reg_results = perform_regression_analysis(X, y, model_type)
                            
                            # Store model in session state
                            st.session_state.model = reg_results['model']
                            st.session_state.feature_names = feature_cols
                            
                            # Tree analysis
                            tree_results = perform_tree_analysis(X, y, use_random_forest)
                            
                            # PCA analysis
                            pca_results = perform_pca_analysis(X)
                            
                            # Optional analyses
                            vif_results = None
                            if perform_vif:
                                vif_results = calculate_vif(X)
                            
                            outliers_results = None
                            if detect_outliers_flag:
                                outliers_results = detect_outliers(df, feature_cols)
                            
                            # Store results
                            st.session_state.analysis_results = AnalysisResults(
                                correlation_matrix=corr_matrix,
                                regression_results=reg_results,
                                tree_results=tree_results,
                                pca_results=pca_results,
                                vif_results=vif_results,
                                outliers=outliers_results
                            )
                            
                            # Update analysis summary
                            analysis_summary["回帰分析"] = {
                                "R²スコア（テスト）": f"{reg_results['test_r2']:.4f}",
                                "RMSE": f"{reg_results['test_rmse']:.4f}",
                                "交差検証R²": f"{reg_results['cv_mean']:.4f} ± {reg_results['cv_std']:.4f}"
                            }
                            
                            analysis_summary["決定木分析"] = {
                                "R²スコア": f"{tree_results['test_r2']:.4f}",
                                "RMSE": f"{tree_results['test_rmse']:.4f}"
                            }
                            
                            st.success("✅ 分析が完了しました！")
                    
                    # Display results if available
                    if st.session_state.analysis_results is not None:
                        results = st.session_state.analysis_results
                        
                        # Create tabs
                        tabs = st.tabs([
                            "📊 相関分析",
                            "📈 回帰分析",
                            "🌳 決定木分析",
                            "🎯 主成分分析",
                            "⚠️ 診断",
                            "📚 統計ガイド",
                            "🔮 What-If",
                            "🤖 AI連携"
                        ])
                        
                        # Tab 1: Correlation Analysis
                        with tabs[0]:
                            st.header("相関分析")
                            fig_corr = create_correlation_heatmap(results.correlation_matrix)
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            # Correlation with target
                            st.subheader(f"「{target_col}」との相関")
                            target_corr = results.correlation_matrix[target_col].sort_values(ascending=False)
                            st.dataframe(target_corr[target_corr.index != target_col])
                        
                        # Tab 2: Regression Analysis
                        with tabs[1]:
                            st.header("回帰分析")
                            
                            # Metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("R² (テスト)", f"{results.regression_results['test_r2']:.4f}")
                            with col2:
                                st.metric("RMSE", f"{results.regression_results['test_rmse']:.4f}")
                            with col3:
                                st.metric("MAE", f"{results.regression_results['test_mae']:.4f}")
                            with col4:
                                st.metric("CV R²", f"{results.regression_results['cv_mean']:.4f}")
                            
                            # Plots
                            fig_reg = create_regression_plots(results.regression_results)
                            st.plotly_chart(fig_reg, use_container_width=True)
                            
                            # Feature importance
                            if results.regression_results['feature_importance'] is not None:
                                st.subheader("回帰係数")
                                fig_coef = create_feature_importance_chart(
                                    results.regression_results['feature_importance']
                                )
                                st.plotly_chart(fig_coef, use_container_width=True)
                            
                            # Statistical summary
                            if 'model_sm' in results.regression_results:
                                with st.expander("詳細な統計情報"):
                                    st.text(str(results.regression_results['model_sm'].summary()))
                        
                        # Tab 3: Decision Tree Analysis
                        with tabs[2]:
                            st.header("決定木分析" + ("（ランダムフォレスト）" if use_random_forest else ""))
                            
                            # Metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("R² (テスト)", f"{results.tree_results['test_r2']:.4f}")
                            with col2:
                                st.metric("RMSE", f"{results.tree_results['test_rmse']:.4f}")
                            
                            # Feature importance
                            st.subheader("特徴量重要度")
                            fig_tree = create_feature_importance_chart(
                                results.tree_results['feature_importance']
                            )
                            st.plotly_chart(fig_tree, use_container_width=True)
                            
                            # Importance table
                            st.dataframe(results.tree_results['feature_importance'])
                        
                        # Tab 4: PCA Analysis
                        with tabs[3]:
                            st.header("主成分分析（PCA）")
                            
                            # Variance explained
                            fig_scree, fig_biplot = create_pca_plots(results.pca_results)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.plotly_chart(fig_scree, use_container_width=True)
                            with col2:
                                st.plotly_chart(fig_biplot, use_container_width=True)
                            
                            # Loadings
                            st.subheader("主成分負荷量")
                            st.dataframe(results.pca_results['loadings'])
                        
                        # Tab 5: Diagnostics
                        with tabs[4]:
                            st.header("診断")
                            
                            # VIF Analysis
                            if results.vif_results is not None:
                                st.subheader("多重共線性チェック（VIF）")
                                st.dataframe(results.vif_results)
                                
                                # Warning for high VIF
                                high_vif = results.vif_results[results.vif_results['VIF'] > 10]
                                if not high_vif.empty:
                                    st.warning(f"⚠️ 以下の変数で多重共線性が検出されました: {', '.join(high_vif['Variable'].tolist())}")
                            
                            # Outlier Detection
                            if results.outliers is not None:
                                st.subheader("外れ値検出")
                                
                                outlier_summary = []
                                for col, outlier_info in results.outliers.items():
                                    iqr_count = len(outlier_info['iqr_outliers'])
                                    z_count = len(outlier_info['z_outliers'])
                                    outlier_summary.append({
                                        '変数': col,
                                        'IQR外れ値': iqr_count,
                                        'Z-score外れ値': z_count,
                                        '下限': f"{outlier_info['lower_bound']:.2f}",
                                        '上限': f"{outlier_info['upper_bound']:.2f}"
                                    })
                                
                                st.dataframe(pd.DataFrame(outlier_summary))
                        
                        # Tab 6: Statistics Guide
                        with tabs[5]:
                            show_statistics_guide()
                        
                        # Tab 7: What-If Simulation
                        with tabs[6]:
                            st.header("🔮 What-Ifシミュレーション")
                            
                            if st.session_state.model is not None:
                                st.info("変数の値を調整して、予測値への影響を確認できます")
                                
                                # Get base values
                                base_values = {col: df[col].mean() for col in feature_cols}
                                adjusted_values = base_values.copy()
                                
                                # Variable adjustments
                                st.subheader("変数の調整")
                                
                                # Create columns for sliders
                                n_cols = 2
                                cols = st.columns(n_cols)
                                
                                for i, var in enumerate(feature_cols):
                                    with cols[i % n_cols]:
                                        var_min = df[var].min()
                                        var_max = df[var].max()
                                        var_mean = df[var].mean()
                                        
                                        st.write(f"**{var}**")
                                        adjusted_values[var] = st.slider(
                                            f"",
                                            min_value=float(var_min),
                                            max_value=float(var_max),
                                            value=float(var_mean),
                                            step=float((var_max - var_min) / 100),
                                            key=f"whatif_slider_{var}"
                                        )
                                        
                                        # Show change
                                        change_pct = (adjusted_values[var] - base_values[var]) / base_values[var] * 100
                                        if abs(change_pct) > 0.1:
                                            st.caption(f"変化: {change_pct:+.1f}%")
                                
                                # Run simulation
                                if st.button("シミュレーション実行", type="primary"):
                                    sim_results = perform_what_if_simulation(
                                        st.session_state.model,
                                        feature_cols,
                                        base_values,
                                        adjusted_values
                                    )
                                    
                                    # Display results
                                    st.markdown("---")
                                    st.subheader("シミュレーション結果")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "ベース予測値",
                                            f"{sim_results['base_prediction']:.4f}"
                                        )
                                    with col2:
                                        st.metric(
                                            "調整後予測値",
                                            f"{sim_results['adjusted_prediction']:.4f}",
                                            f"{sim_results['change']:+.4f}"
                                        )
                                    with col3:
                                        st.metric(
                                            "変化率",
                                            f"{sim_results['change_percent']:+.2f}%"
                                        )
                                    
                                    # Sensitivity analysis
                                    st.subheader("感度分析（10%増加時の影響）")
                                    
                                    sensitivity_df = pd.DataFrame([
                                        {"変数": k, "影響度(%)": v}
                                        for k, v in sim_results['sensitivity'].items()
                                    ]).sort_values("影響度(%)", key=abs, ascending=False)
                                    
                                    fig_sensitivity = px.bar(
                                        sensitivity_df,
                                        x="影響度(%)",
                                        y="変数",
                                        orientation='h',
                                        title="各変数の感度（10%増加時の予測値変化率）"
                                    )
                                    st.plotly_chart(fig_sensitivity, use_container_width=True)
                            else:
                                st.warning("先に分析を実行してください")
                        
                        # Tab 8: AI Integration
                        with tabs[7]:
                            st.header("🤖 AI連携（プロンプト生成）")
                            
                            if st.session_state.analysis_results is not None:
                                st.info("分析結果を解釈するためのAIプロンプトを生成します")
                                
                                # Settings
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    interpretation_level = st.selectbox(
                                        "解釈レベル",
                                        ["初心者", "一般", "専門家"],
                                        index=1,
                                        help="説明の詳細度を選択してください"
                                    )
                                
                                with col2:
                                    business_context = st.text_input(
                                        "ビジネスコンテキスト（任意）",
                                        placeholder="例：医薬品の市場シェア分析",
                                        help="具体的な業界や状況を入力すると、より適切な解釈が得られます"
                                    )
                                
                                # Generate prompt button
                                if st.button("📝 解釈プロンプトを生成", type="primary"):
                                    # Prepare analysis summary
                                    analysis_summary = {
                                        "データ概要": {
                                            "サンプル数": len(X),
                                            "説明変数": feature_cols,
                                            "ターゲット変数": target_col
                                        },
                                        "相関分析": {
                                            "最大相関": float(results.correlation_matrix[target_col].abs().sort_values(ascending=False).iloc[1]),
                                            "上位相関変数": results.correlation_matrix[target_col].abs().sort_values(ascending=False).index[1:4].tolist()
                                        },
                                        "回帰分析": {
                                            "R²スコア": results.regression_results['test_r2'],
                                            "RMSE": results.regression_results['test_rmse'],
                                            "交差検証R²": results.regression_results['cv_mean']
                                        }
                                    }
                                    
                                    if results.tree_results:
                                        analysis_summary["決定木分析"] = {
                                            "R²スコア": results.tree_results['test_r2'],
                                            "重要度上位": results.tree_results['feature_importance'].head(3)['Variable'].tolist()
                                        }
                                    
                                    # Generate prompt
                                    prompt = generate_analysis_prompt(
                                        analysis_summary,
                                        business_context,
                                        interpretation_level
                                    )
                                    
                                    # Display prompt
                                    st.subheader("📄 生成されたプロンプト")
                                    
                                    st.text_area(
                                        "以下のプロンプトをコピーしてAIに貼り付けてください",
                                        value=prompt,
                                        height=400,
                                        key="generated_prompt_optimized"
                                    )
                                    
                                    # Copy instructions
                                    with st.expander("💡 使い方"):
                                        st.markdown("""
                                        ### ChatGPT
                                        1. https://chat.openai.com にアクセス
                                        2. プロンプトを貼り付けて送信
                                        
                                        ### Claude
                                        1. https://claude.ai にアクセス
                                        2. プロンプトを貼り付けて送信
                                        
                                        ### Google Gemini
                                        1. https://gemini.google.com にアクセス
                                        2. プロンプトを貼り付けて送信
                                        """)
                                    
                                    # Download button
                                    st.download_button(
                                        label="💾 プロンプトをダウンロード",
                                        data=prompt,
                                        file_name="analysis_prompt.txt",
                                        mime="text/plain"
                                    )
                            else:
                                st.warning("先に分析を実行してください")
                
                else:
                    st.warning("説明変数を選択してください")
            else:
                st.error("数値列が見つかりません。CSVファイルに数値データが含まれているか確認してください。")
        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome message
        st.info("👈 左側のサイドバーからCSVファイルをアップロードして開始してください")
        
        # Sample data generation
        with st.expander("📝 サンプルデータを生成"):
            if st.button("サンプルCSVを生成"):
                sample_data = pd.DataFrame({
                    '年月': pd.date_range('2024-01', periods=12, freq='ME').strftime('%Y-%m'),
                    'シェア': np.random.uniform(0.1, 0.5, 12),
                    '売上': np.random.uniform(100000, 500000, 12),
                    '広告費': np.random.uniform(10000, 50000, 12),
                    '店舗数': np.random.randint(10, 50, 12),
                    '顧客満足度': np.random.uniform(3.0, 5.0, 12)
                })
                
                csv = sample_data.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 サンプルCSVをダウンロード",
                    data=csv,
                    file_name="sample_share_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()