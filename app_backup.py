import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import io
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="製品シェア要因分析ツール",
    page_icon="📊",
    layout="wide"
)

st.title("📊 製品シェア要因分析ツール")
st.markdown("---")

# サイドバー
with st.sidebar:
    st.header("📁 データアップロード")
    
    # ファイルアップローダー
    uploaded_file = st.file_uploader(
        "CSVファイルを選択してください",
        type=['csv'],
        help="分析したいCSVファイルをアップロードしてください"
    )
    
    # エンコーディング選択
    encoding = st.selectbox(
        "文字エンコーディング",
        ["utf-8", "shift-jis", "cp932", "utf-8-sig"],
        index=1,
        help="日本語のCSVファイルの場合は'shift-jis'を選択してください"
    )
    
    st.markdown("---")
    st.info(
        "💡 **使い方**\n"
        "1. CSVファイルをアップロード\n"
        "2. ターゲット変数を選択\n"
        "3. 説明変数を選択\n"
        "4. 分析を実行"
    )

# メインコンテンツ
if uploaded_file is not None:
    try:
        # データ読み込み
        df = pd.read_csv(uploaded_file, encoding=encoding)
        
        # パーセンテージ文字列を数値に変換
        for col in df.columns:
            if df[col].dtype == 'object':
                if df[col].astype(str).str.contains('%').any():
                    df[col] = df[col].str.replace('%', '').replace('-', '0').astype(float)
        
        # データ概要を表示
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("行数", f"{len(df):,}")
        with col2:
            st.metric("列数", f"{len(df.columns):,}")
        with col3:
            st.metric("欠損値", f"{df.isnull().sum().sum():,}")
        
        # データプレビュー
        with st.expander("📋 データプレビュー", expanded=False):
            st.dataframe(df.head(10))
            
            # 基本統計量
            st.subheader("基本統計量")
            st.dataframe(df.describe())
        
        st.markdown("---")
        
        # 分析設定
        st.header("⚙️ 分析設定")
        
        # 数値列のみを抽出
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # ターゲット変数の選択
                target_col = st.selectbox(
                    "🎯 ターゲット変数（目的変数）",
                    numeric_cols,
                    help="予測・分析したい変数を選択してください（例：製品シェア）"
                )
            
            with col2:
                # 説明変数の選択
                available_features = [col for col in numeric_cols if col != target_col]
                feature_cols = st.multiselect(
                    "📊 説明変数（独立変数）",
                    available_features,
                    default=available_features[:min(10, len(available_features))],
                    help="ターゲット変数に影響を与える可能性のある変数を選択してください"
                )
            
            if st.button("🔍 分析を実行", type="primary", use_container_width=True):
                if len(feature_cols) > 0:
                    # プログレスバー
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # タブを作成
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
                        ["📈 相関分析", "📉 回帰分析", "🌀 主成分分析", "🌳 決定木分析", "📝 レポート", "📚 統計ガイド", "🎯 What-Ifシミュレーション", "🤖 AI分析プロンプト"]
                    )
                    
                    # データ準備
                    data_for_analysis = df[[target_col] + feature_cols].dropna()
                    X = data_for_analysis[feature_cols]
                    y = data_for_analysis[target_col]
                    
                    # 初期化（他のタブで使用される変数）
                    model = None
                    model_sm = None
                    corr_df = None
                    coef_df = None
                    importance_df = None
                    test_r2 = None
                    test_rmse = None
                    tree_r2 = None
                    
                    # 1. 相関分析
                    with tab1:
                        status_text.text("相関分析を実行中...")
                        progress_bar.progress(20)
                        
                        st.header("📈 相関分析")
                        
                        # 相関係数を計算
                        correlations = []
                        for col in feature_cols:
                            corr, p_value = stats.pearsonr(X[col], y)
                            correlations.append({
                                '変数': col,
                                '相関係数': corr,
                                'p値': p_value,
                                '有意性': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                            })
                        
                        corr_df = pd.DataFrame(correlations)
                        corr_df = corr_df.sort_values('相関係数', key=abs, ascending=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 相関係数のバーチャート
                            fig = px.bar(
                                corr_df.head(15),
                                x='相関係数',
                                y='変数',
                                orientation='h',
                                color='相関係数',
                                color_continuous_scale='RdBu_r',
                                color_continuous_midpoint=0,
                                title=f'{target_col}との相関係数（上位15変数）'
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # 相関行列のヒートマップ
                            corr_matrix = data_for_analysis.corr()
                            fig = px.imshow(
                                corr_matrix,
                                text_auto='.2f',
                                aspect='auto',
                                color_continuous_scale='RdBu_r',
                                color_continuous_midpoint=0,
                                title='相関行列ヒートマップ'
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 相関分析結果の表
                        st.subheader("📊 相関分析結果")
                        st.dataframe(
                            corr_df.style.format({
                                '相関係数': '{:.4f}',
                                'p値': '{:.6f}'
                            }).background_gradient(subset=['相関係数'], cmap='RdBu_r', vmin=-1, vmax=1),
                            use_container_width=True
                        )
                    
                    # 2. 回帰分析
                    with tab2:
                        status_text.text("回帰分析を実行中...")
                        progress_bar.progress(40)
                        
                        st.header("📉 回帰分析")
                        
                        # データ分割
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )
                        
                        # 重回帰分析
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        
                        y_pred_train = model.predict(X_train)
                        y_pred_test = model.predict(X_test)
                        
                        train_r2 = r2_score(y_train, y_pred_train)
                        test_r2 = r2_score(y_test, y_pred_test)
                        
                        # メトリクス表示
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("訓練データ R²", f"{train_r2:.4f}")
                        with col2:
                            st.metric("テストデータ R²", f"{test_r2:.4f}")
                        with col3:
                            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                            st.metric("訓練データ RMSE", f"{train_rmse:.4f}")
                        with col4:
                            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                            st.metric("テストデータ RMSE", f"{test_rmse:.4f}")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 回帰係数
                            coef_df = pd.DataFrame({
                                '変数': feature_cols,
                                '回帰係数': model.coef_,
                                '影響度（絶対値）': np.abs(model.coef_)
                            }).sort_values('影響度（絶対値）', ascending=False)
                            
                            fig = px.bar(
                                coef_df.head(15),
                                x='回帰係数',
                                y='変数',
                                orientation='h',
                                color='回帰係数',
                                color_continuous_scale='RdBu_r',
                                color_continuous_midpoint=0,
                                title='回帰係数（上位15変数）'
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # 予測精度の散布図
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=y_test,
                                y=y_pred_test,
                                mode='markers',
                                name='予測値',
                                marker=dict(color='blue', opacity=0.5)
                            ))
                            fig.add_trace(go.Scatter(
                                x=[y_test.min(), y_test.max()],
                                y=[y_test.min(), y_test.max()],
                                mode='lines',
                                name='理想線',
                                line=dict(color='red', dash='dash')
                            ))
                            fig.update_layout(
                                title=f'予測精度 (R²={test_r2:.3f})',
                                xaxis_title='実測値',
                                yaxis_title='予測値',
                                height=500
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # statsmodelsによる詳細分析
                        X_sm = sm.add_constant(X)
                        model_sm = sm.OLS(y, X_sm).fit()
                        
                        st.subheader("📊 統計的検定結果")
                        summary_data = pd.DataFrame({
                            '変数': ['定数項'] + feature_cols,
                            '係数': model_sm.params.values,
                            '標準誤差': model_sm.bse.values,
                            't値': model_sm.tvalues.values,
                            'p値': model_sm.pvalues.values,
                            '有意性': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '' 
                                    for p in model_sm.pvalues.values]
                        })
                        
                        st.dataframe(
                            summary_data.style.format({
                                '係数': '{:.4f}',
                                '標準誤差': '{:.4f}',
                                't値': '{:.4f}',
                                'p値': '{:.6f}'
                            }),
                            use_container_width=True
                        )
                        
                        st.info(f"調整済みR²: {model_sm.rsquared_adj:.4f} | F統計量: {model_sm.fvalue:.4f} | F値のp値: {model_sm.f_pvalue:.6f}")
                    
                    # 3. 主成分分析
                    with tab3:
                        status_text.text("主成分分析を実行中...")
                        progress_bar.progress(60)
                        
                        st.header("🌀 主成分分析（PCA）")
                        
                        # データの標準化
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # PCA実行
                        pca = PCA()
                        pca.fit(X_scaled)
                        
                        # 寄与率
                        explained_var = pca.explained_variance_ratio_
                        cumsum_var = np.cumsum(explained_var)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # スクリープロット
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=list(range(1, min(11, len(explained_var)+1))),
                                y=explained_var[:10],
                                name='個別寄与率',
                                marker_color='lightblue'
                            ))
                            fig.add_trace(go.Scatter(
                                x=list(range(1, min(11, len(cumsum_var)+1))),
                                y=cumsum_var[:10],
                                name='累積寄与率',
                                mode='lines+markers',
                                marker_color='red'
                            ))
                            fig.update_layout(
                                title='主成分の寄与率',
                                xaxis_title='主成分番号',
                                yaxis_title='寄与率',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # 主成分負荷量
                            n_components = min(3, len(feature_cols))
                            pca_reduced = PCA(n_components=n_components)
                            X_pca = pca_reduced.fit_transform(X_scaled)
                            
                            loadings = pd.DataFrame(
                                pca_reduced.components_.T,
                                index=feature_cols,
                                columns=[f'PC{i+1}' for i in range(n_components)]
                            )
                            
                            fig = px.imshow(
                                loadings.T,
                                text_auto='.2f',
                                aspect='auto',
                                color_continuous_scale='RdBu_r',
                                color_continuous_midpoint=0,
                                title='主成分負荷量'
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # 累積寄与率80%を達成する主成分数
                        n_components_80 = np.argmax(cumsum_var >= 0.8) + 1
                        st.info(f"累積寄与率80%を達成する主成分数: {n_components_80}")
                        
                        # 主成分の解釈
                        st.subheader("主成分の解釈")
                        for i in range(min(3, pca_reduced.n_components_)):
                            st.write(f"**第{i+1}主成分（寄与率: {explained_var[i]:.1%}）**")
                            important_vars = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5)
                            interpretation_df = pd.DataFrame({
                                '変数': important_vars.index,
                                '負荷量': [loadings.loc[var, f'PC{i+1}'] for var in important_vars.index]
                            })
                            st.dataframe(interpretation_df, use_container_width=True)
                    
                    # 4. 決定木分析
                    with tab4:
                        status_text.text("決定木分析を実行中...")
                        progress_bar.progress(80)
                        
                        st.header("🌳 決定木分析")
                        
                        # 決定木モデル
                        tree_model = DecisionTreeRegressor(
                            max_depth=4,
                            min_samples_split=20,
                            min_samples_leaf=10,
                            random_state=42
                        )
                        tree_model.fit(X_train, y_train)
                        
                        y_pred_tree = tree_model.predict(X_test)
                        tree_r2 = r2_score(y_test, y_pred_tree)
                        
                        # 変数重要度
                        importance_df = pd.DataFrame({
                            '変数': feature_cols,
                            '重要度': tree_model.feature_importances_
                        }).sort_values('重要度', ascending=False)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 変数重要度のバーチャート
                            fig = px.bar(
                                importance_df.head(15),
                                x='重要度',
                                y='変数',
                                orientation='h',
                                title='変数の重要度（上位15）'
                            )
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # モデル性能メトリクス
                            st.metric("テストデータ R²", f"{tree_r2:.4f}")
                            
                            # 重要な変数のリスト
                            st.subheader("最も重要な変数")
                            for i, row in importance_df.head(5).iterrows():
                                st.write(f"{i+1}. **{row['変数']}**: {row['重要度']:.4f}")
                    
                    # 5. レポート
                    with tab5:
                        status_text.text("レポートを生成中...")
                        progress_bar.progress(100)
                        
                        st.header("📝 分析レポート")
                        
                        # サマリー
                        st.subheader("分析サマリー")
                        
                        summary_text = f"""
                        ### 分析概要
                        - **ターゲット変数**: {target_col}
                        - **説明変数数**: {len(feature_cols)}
                        - **サンプル数**: {len(data_for_analysis)}
                        
                        ### 主要な発見
                        
                        #### 1. 相関分析
                        - 最も強い相関を持つ変数: **{corr_df.iloc[0]['変数']}** (r={corr_df.iloc[0]['相関係数']:.3f})
                        - 有意な相関を持つ変数数: {len(corr_df[corr_df['有意性'] != ''])}
                        
                        #### 2. 回帰分析
                        - モデルの決定係数 (R²): {test_r2:.4f}
                        - 調整済みR²: {model_sm.rsquared_adj:.4f}
                        - 最も影響度の高い変数: **{coef_df.iloc[0]['変数']}**
                        
                        #### 3. 主成分分析
                        - 累積寄与率80%達成に必要な主成分数: {n_components_80}
                        - 第1主成分の寄与率: {explained_var[0]:.1%}
                        
                        #### 4. 決定木分析
                        - モデルの決定係数 (R²): {tree_r2:.4f}
                        - 最も重要な変数: **{importance_df.iloc[0]['変数']}** (重要度: {importance_df.iloc[0]['重要度']:.3f})
                        """
                        
                        st.markdown(summary_text)
                        
                        # 推奨アクション
                        st.subheader("推奨アクション")
                        recommendations = []
                        
                        # 相関が強い変数がある場合
                        if abs(corr_df.iloc[0]['相関係数']) > 0.5:
                            recommendations.append(
                                f"- {corr_df.iloc[0]['変数']}は{target_col}と強い相関があります。"
                                f"この変数の改善に注力することを推奨します。"
                            )
                        
                        # 回帰モデルの精度が高い場合
                        if test_r2 > 0.7:
                            recommendations.append(
                                f"- 回帰モデルの精度が高い（R²={test_r2:.3f}）ため、"
                                f"予測モデルとして活用できます。"
                            )
                        
                        # 決定木で重要な変数
                        if importance_df.iloc[0]['重要度'] > 0.3:
                            recommendations.append(
                                f"- {importance_df.iloc[0]['変数']}が決定木分析で最も重要な変数として"
                                f"特定されました。優先的に管理することを推奨します。"
                            )
                        
                        if recommendations:
                            for rec in recommendations:
                                st.write(rec)
                        else:
                            st.write("- さらに詳細な分析のため、追加データの収集を検討してください。")
                        
                        # レポートのダウンロード
                        st.subheader("レポートのエクスポート")
                        
                        # レポート内容を文字列として生成
                        report_content = f"""
製品シェア要因分析レポート
{'='*60}

ターゲット変数: {target_col}
説明変数数: {len(feature_cols)}
サンプル数: {len(data_for_analysis)}

【相関分析の要約】
有意な相関を持つ変数数: {len(corr_df[corr_df['有意性'] != ''])}
最も強い相関を持つ変数TOP3:
{corr_df.head(3).to_string()}

【回帰分析の要約】
調整済みR²: {model_sm.rsquared_adj:.4f}
テストデータR²: {test_r2:.4f}

【主成分分析の要約】
累積寄与率80%達成に必要な主成分数: {n_components_80}
第1主成分の寄与率: {explained_var[0]:.1%}

【決定木分析の要約】
テストデータR²: {tree_r2:.4f}
重要度の高い変数TOP3:
{importance_df.head(3).to_string()}

【推奨アクション】
{chr(10).join(recommendations) if recommendations else 'さらに詳細な分析のため、追加データの収集を検討してください。'}
                        """
                        
                        st.download_button(
                            label="📥 レポートをダウンロード",
                            data=report_content,
                            file_name="share_analysis_report.txt",
                            mime="text/plain"
                        )
                    
                    # 6. 統計ガイド
                    with tab6:
                        st.header("📚 統計ガイド")
                        st.markdown("統計手法の解説と結果の解釈方法")
                        
                        # サブタブを作成
                        guide_tab1, guide_tab2, guide_tab3, guide_tab4, guide_tab5 = st.tabs([
                            "相関分析", "回帰分析", "多重共線性", "交差検証", "解釈のコツ"
                        ])
                        
                        with guide_tab1:
                            st.subheader("📈 相関分析の解釈")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**相関係数の判定基準**")
                                corr_guide_df = pd.DataFrame({
                                    '相関係数の絶対値': ['0.9〜1.0', '0.7〜0.9', '0.5〜0.7', '0.3〜0.5', '0.0〜0.3'],
                                    '関係の強さ': ['非常に強い', '強い', '中程度', '弱い', 'ほとんどなし'],
                                    '実務的意味': ['ほぼ完全な線形関係', '明確な関連性', '一定の関連性', 'わずかな関連性', '実質的に無関係']
                                })
                                st.dataframe(corr_guide_df, use_container_width=True)
                            
                            with col2:
                                st.markdown("**相関の種類**")
                                st.info("""
                                **Pearson相関係数**
                                - 線形関係の強さを測定
                                - 正規分布を仮定
                                - 外れ値の影響を受けやすい
                                
                                **Spearman相関係数**
                                - 順位相関（単調関係）を測定
                                - 正規分布を仮定しない
                                - 外れ値の影響を受けにくい
                                """)
                            
                            st.warning("""
                            ⚠️ **注意点**
                            - 相関は因果関係を意味しない
                            - 第3の変数（交絡因子）の影響を考慮する必要がある
                            - 非線形関係は検出できない
                            """)
                        
                        with guide_tab2:
                            st.subheader("📉 回帰分析の評価指標")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**R²（決定係数）**")
                                st.success("""
                                モデルが説明できる変動の割合
                                - 0.8以上: 優秀 ✅
                                - 0.5〜0.8: 良好 ⚠️
                                - 0.5未満: 改善必要 ❌
                                """)
                                
                                st.markdown("**p値**")
                                st.success("""
                                係数が0である確率
                                - p < 0.001: *** (非常に有意)
                                - p < 0.01: ** (有意)
                                - p < 0.05: * (やや有意)
                                - p ≥ 0.05: 有意でない
                                """)
                            
                            with col2:
                                st.markdown("**調整済みR²**")
                                st.info("""
                                変数の数を考慮した決定係数
                                - 変数を増やしても自動的に上がらない
                                - モデル比較に適している
                                - 通常のR²より小さい値
                                """)
                                
                                st.markdown("**RMSE**")
                                st.info("""
                                予測誤差の標準偏差
                                - 元のデータと同じ単位
                                - 小さいほど良い
                                - 外れ値の影響を受けやすい
                                """)
                            
                            st.markdown("**回帰係数の解釈例**")
                            st.code("""
                            営業人員数の係数 = 0.35
                            → 営業人員が1人増えると、製品Aシェアが0.35％増加
                            （他の変数が一定の場合）
                            """)
                        
                        with guide_tab3:
                            st.subheader("🔍 多重共線性（VIF）")
                            
                            st.markdown("**VIF（分散拡大係数）の判定基準**")
                            vif_guide_df = pd.DataFrame({
                                'VIF値': ['VIF < 5', '5 ≤ VIF < 10', 'VIF ≥ 10'],
                                '判定': ['問題なし ✅', '中程度の多重共線性 ⚠️', '深刻な多重共線性 ❌'],
                                '対処法': [
                                    'そのまま使用可能',
                                    '変数の削除を検討',
                                    '変数削除、主成分分析、Ridge回帰'
                                ]
                            })
                            st.dataframe(vif_guide_df, use_container_width=True)
                            
                            st.info("""
                            **計算式**: VIF = 1 / (1 - R²ᵢ)
                            
                            R²ᵢ: 変数iを他の変数で回帰した時の決定係数
                            """)
                            
                            st.markdown("**外れ値の検出方法**")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**IQR法**")
                                st.code("""
                                下限 = Q1 - 1.5 × IQR
                                上限 = Q3 + 1.5 × IQR
                                (IQR = Q3 - Q1)
                                """)
                            with col2:
                                st.markdown("**Z-score法**")
                                st.code("""
                                Z = (x - μ) / σ
                                |Z| > 3 → 外れ値
                                """)
                        
                        with guide_tab4:
                            st.subheader("🔄 交差検証")
                            
                            st.markdown("**K-fold交差検証の仕組み**")
                            st.info("""
                            1. データをK個に分割
                            2. K-1個で学習、1個で検証
                            3. これをK回繰り返す
                            4. K回の結果を平均化
                            """)
                            
                            st.markdown("**結果の解釈**")
                            st.success("""
                            例: R² = 0.75 (±0.08)
                            - 平均R²: 0.75（モデルの性能）
                            - 標準偏差: 0.08（安定性）
                            - 標準偏差が小さい = 安定したモデル
                            """)
                            
                            st.markdown("**正則化回帰**")
                            regularization_df = pd.DataFrame({
                                '手法': ['Ridge (L2)', 'Lasso (L1)', 'Elastic Net'],
                                '特徴': [
                                    '全変数使用、係数縮小',
                                    '変数選択、係数を0に',
                                    'RidgeとLassoの組み合わせ'
                                ],
                                '適用場面': [
                                    '多重共線性がある場合',
                                    '変数選択したい場合',
                                    'バランスを取りたい場合'
                                ]
                            })
                            st.dataframe(regularization_df, use_container_width=True)
                        
                        with guide_tab5:
                            st.subheader("💡 結果解釈のコツ")
                            
                            st.markdown("**総合的な判断例**")
                            st.success("""
                            ✅ 良好なモデルの例:
                            - 調整済みR² = 0.72（良好な説明力）
                            - 交差検証R² = 0.68 (±0.05)（安定した性能）
                            - VIF最大値 = 4.2（多重共線性なし）
                            - Durbin-Watson = 1.95（自己相関なし）
                            """)
                            
                            st.markdown("**よくある誤解**")
                            st.warning("""
                            ❌ **R²が高い = 良いモデル**
                            → 過学習の可能性。交差検証で確認必要
                            
                            ❌ **p値 < 0.05 = 重要な変数**
                            → 効果の大きさも確認。実務的意味を考慮
                            
                            ❌ **相関が高い = 因果関係**
                            → 相関≠因果。理論的背景の検討が必要
                            
                            ❌ **複雑なモデル = 良いモデル**
                            → シンプルで解釈しやすいモデルが実務では有用
                            """)
                            
                            st.markdown("**実務での活用ポイント**")
                            st.info("""
                            1. 単一の指標だけでなく、複数の指標を総合的に判断
                            2. 統計的有意性だけでなく、実務的な意味を考慮
                            3. モデルの前提条件を満たしているか確認
                            4. 予測の不確実性を認識し、適切な意思決定を行う
                            """)
                    
                    # 7. What-Ifシミュレーション
                    with tab7:
                        st.header("🎯 What-Ifシミュレーション")
                        st.markdown("変数を調整して、ターゲット変数への影響をリアルタイムで確認できます")
                        
                        # 回帰モデルを使用（tab2で作成済み）
                        if 'model' in locals():
                            # シミュレーション設定
                            st.subheader("📊 シミュレーション設定")
                            
                            # 現在の値を取得
                            current_values = X.mean().to_dict()
                            
                            # シナリオ管理
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                scenario_name = st.text_input("シナリオ名", value="シナリオ1")
                            with col2:
                                baseline_pred = model.predict(X.mean().values.reshape(1, -1))[0]
                                st.metric("ベースライン予測値", f"{baseline_pred:.2f}")
                            with col3:
                                save_scenario = st.button("シナリオ保存", type="secondary")
                            
                            # スライダーで変数を調整
                            st.subheader("🎚️ 変数の調整")
                            
                            # 調整する変数を選択
                            selected_vars = st.multiselect(
                                "調整する変数を選択",
                                feature_cols,
                                default=feature_cols[:min(5, len(feature_cols))],
                                help="スライダーで調整したい変数を選択してください"
                            )
                            
                            # 各変数のスライダーを作成
                            adjusted_values = current_values.copy()
                            
                            if selected_vars:
                                # スライダーを2列で表示
                                cols = st.columns(2)
                                for i, var in enumerate(selected_vars):
                                    with cols[i % 2]:
                                        # 変数の統計量を取得
                                        var_min = X[var].min()
                                        var_max = X[var].max()
                                        var_mean = X[var].mean()
                                        var_std = X[var].std()
                                        
                                        # スライダーの範囲を設定（平均±3標準偏差）
                                        slider_min = max(var_min, var_mean - 3 * var_std)
                                        slider_max = min(var_max, var_mean + 3 * var_std)
                                        
                                        # 現在の値を表示
                                        st.markdown(f"**{var}**")
                                        col_a, col_b = st.columns([3, 1])
                                        with col_a:
                                            adjusted_values[var] = st.slider(
                                                f"",
                                                min_value=float(slider_min),
                                                max_value=float(slider_max),
                                                value=float(var_mean),
                                                step=float((slider_max - slider_min) / 100),
                                                key=f"slider_{var}"
                                            )
                                        with col_b:
                                            change_pct = ((adjusted_values[var] - var_mean) / var_mean) * 100
                                            if change_pct > 0:
                                                st.markdown(f"<span style='color: green'>+{change_pct:.1f}%</span>", unsafe_allow_html=True)
                                            elif change_pct < 0:
                                                st.markdown(f"<span style='color: red'>{change_pct:.1f}%</span>", unsafe_allow_html=True)
                                            else:
                                                st.markdown(f"{change_pct:.1f}%")
                                
                                # 予測値の計算
                                st.markdown("---")
                                
                                # 調整後の値で予測
                                adjusted_array = np.array([adjusted_values[col] for col in feature_cols]).reshape(1, -1)
                                new_prediction = model.predict(adjusted_array)[0]
                                
                                # 結果の表示
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "調整後の予測値",
                                        f"{new_prediction:.2f}",
                                        f"{new_prediction - baseline_pred:.2f}"
                                    )
                                
                                with col2:
                                    change_pct = ((new_prediction - baseline_pred) / baseline_pred) * 100
                                    st.metric(
                                        "変化率",
                                        f"{change_pct:.1f}%",
                                        f"{abs(change_pct):.1f}%"
                                    )
                                
                                with col3:
                                    # 影響度の判定
                                    if abs(change_pct) < 5:
                                        impact = "低"
                                        color = "🟢"
                                    elif abs(change_pct) < 15:
                                        impact = "中"
                                        color = "🟡"
                                    else:
                                        impact = "高"
                                        color = "🔴"
                                    st.metric("影響度", f"{color} {impact}")
                                
                                # 感度分析
                                st.subheader("📈 感度分析")
                                
                                sensitivity_results = []
                                for var in selected_vars:
                                    # 各変数を±10%変化させた時の影響
                                    test_values = adjusted_values.copy()
                                    original_val = X[var].mean()
                                    
                                    # +10%
                                    test_values[var] = original_val * 1.1
                                    test_array = np.array([test_values[col] for col in feature_cols]).reshape(1, -1)
                                    pred_plus = model.predict(test_array)[0]
                                    
                                    # -10%
                                    test_values[var] = original_val * 0.9
                                    test_array = np.array([test_values[col] for col in feature_cols]).reshape(1, -1)
                                    pred_minus = model.predict(test_array)[0]
                                    
                                    # 感度を計算
                                    sensitivity = (pred_plus - pred_minus) / (0.2 * original_val)
                                    sensitivity_results.append({
                                        '変数': var,
                                        '感度': sensitivity,
                                        '+10%時の予測': pred_plus,
                                        '-10%時の予測': pred_minus,
                                        '影響範囲': pred_plus - pred_minus
                                    })
                                
                                sensitivity_df = pd.DataFrame(sensitivity_results)
                                sensitivity_df = sensitivity_df.sort_values('感度', key=abs, ascending=False)
                                
                                # 感度分析の可視化
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # 感度のバーチャート
                                    fig = px.bar(
                                        sensitivity_df,
                                        x='感度',
                                        y='変数',
                                        orientation='h',
                                        color='感度',
                                        color_continuous_scale='RdBu_r',
                                        color_continuous_midpoint=0,
                                        title='変数の感度（10%変化あたりの影響）'
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # 影響範囲のバーチャート
                                    fig = px.bar(
                                        sensitivity_df,
                                        x='影響範囲',
                                        y='変数',
                                        orientation='h',
                                        title='±10%変化時の予測値の変動幅'
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # 感度分析テーブル
                                st.dataframe(
                                    sensitivity_df.style.format({
                                        '感度': '{:.4f}',
                                        '+10%時の予測': '{:.2f}',
                                        '-10%時の予測': '{:.2f}',
                                        '影響範囲': '{:.2f}'
                                    }).background_gradient(subset=['感度'], cmap='RdBu_r', vmin=-sensitivity_df['感度'].abs().max(), vmax=sensitivity_df['感度'].abs().max()),
                                    use_container_width=True
                                )
                                
                                # 最適化提案
                                st.subheader("🎯 最適化提案")
                                
                                # 目標値設定
                                col1, col2 = st.columns(2)
                                with col1:
                                    target_value = st.number_input(
                                        f"{target_col}の目標値",
                                        value=float(y.mean() * 1.2),
                                        help="達成したい目標値を入力してください"
                                    )
                                
                                with col2:
                                    optimize_btn = st.button("最適化を実行", type="primary")
                                
                                if optimize_btn:
                                    # 簡易的な最適化（感度の高い変数を優先的に調整）
                                    optimized_values = current_values.copy()
                                    remaining_gap = target_value - baseline_pred
                                    
                                    st.info(f"目標達成に必要な改善: {remaining_gap:.2f}")
                                    
                                    recommendations = []
                                    for _, row in sensitivity_df.iterrows():
                                        var = row['変数']
                                        sensitivity = row['感度']
                                        
                                        if sensitivity != 0:
                                            # 必要な変化量を計算
                                            required_change = remaining_gap / sensitivity
                                            current_val = current_values[var]
                                            new_val = current_val + required_change
                                            
                                            # 実現可能な範囲内に制限
                                            var_min = X[var].min()
                                            var_max = X[var].max()
                                            new_val = np.clip(new_val, var_min * 0.8, var_max * 1.2)
                                            
                                            change_pct = ((new_val - current_val) / current_val) * 100
                                            
                                            if abs(change_pct) < 50:  # 50%以内の変化なら推奨
                                                recommendations.append({
                                                    '変数': var,
                                                    '現在値': current_val,
                                                    '推奨値': new_val,
                                                    '変化率': change_pct,
                                                    '期待効果': sensitivity * (new_val - current_val)
                                                })
                                    
                                    if recommendations:
                                        rec_df = pd.DataFrame(recommendations)
                                        rec_df = rec_df.sort_values('期待効果', key=abs, ascending=False)
                                        
                                        st.success("📋 最適化提案")
                                        st.dataframe(
                                            rec_df.head(5).style.format({
                                                '現在値': '{:.2f}',
                                                '推奨値': '{:.2f}',
                                                '変化率': '{:.1f}%',
                                                '期待効果': '{:.2f}'
                                            }),
                                            use_container_width=True
                                        )
                                        
                                        # 提案の要約
                                        top_rec = rec_df.iloc[0]
                                        st.markdown(f"""
                                        **💡 最優先アクション:**
                                        - **{top_rec['変数']}** を **{top_rec['現在値']:.2f}** から **{top_rec['推奨値']:.2f}** に調整
                                        - 期待される効果: {target_col}が **{top_rec['期待効果']:.2f}** 改善
                                        """)
                                
                                # シナリオ比較
                                if 'saved_scenarios' not in st.session_state:
                                    st.session_state.saved_scenarios = []
                                
                                if save_scenario:
                                    # 現在のシナリオを保存
                                    scenario_data = {
                                        'name': scenario_name,
                                        'values': adjusted_values,
                                        'prediction': new_prediction,
                                        'change': new_prediction - baseline_pred
                                    }
                                    st.session_state.saved_scenarios.append(scenario_data)
                                    st.success(f"シナリオ '{scenario_name}' を保存しました")
                                
                                if st.session_state.saved_scenarios:
                                    st.subheader("📊 シナリオ比較")
                                    
                                    scenarios_df = pd.DataFrame(st.session_state.saved_scenarios)
                                    
                                    # シナリオ比較グラフ
                                    fig = px.bar(
                                        scenarios_df,
                                        x='name',
                                        y='prediction',
                                        color='change',
                                        color_continuous_scale='RdYlGn',
                                        title='シナリオ別予測値',
                                        labels={'prediction': '予測値', 'name': 'シナリオ', 'change': '変化量'}
                                    )
                                    fig.add_hline(y=baseline_pred, line_dash="dash", line_color="gray", annotation_text="ベースライン")
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # シナリオテーブル
                                    st.dataframe(
                                        scenarios_df[['name', 'prediction', 'change']].style.format({
                                            'prediction': '{:.2f}',
                                            'change': '{:.2f}'
                                        }),
                                        use_container_width=True
                                    )
                                    
                                    # シナリオクリアボタン
                                    if st.button("シナリオをクリア", type="secondary"):
                                        st.session_state.saved_scenarios = []
                                        st.rerun()
                            else:
                                st.info("調整する変数を選択してください")
                        else:
                            st.warning("先に回帰分析タブでモデルを作成してください")
                    
                    # 8. AI分析プロンプト生成
                    with tab8:
                        st.header("🤖 AI解釈用プロンプト")
                        st.markdown("分析結果をAIに貼り付けて、わかりやすい解釈を得るためのプロンプトを生成します")
                        
                        # フォームを使用してリロードを防ぐ
                        with st.form("prompt_generation_form"):
                            # シンプルな設定
                            st.subheader("📋 基本情報（任意）")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                business_context = st.text_input(
                                    "ビジネスの背景",
                                    placeholder="例：医療機器メーカーのシェア拡大"
                                )
                            with col2:
                                interpretation_level = st.selectbox(
                                    "解釈のレベル",
                                    ["初心者向け（専門用語を避ける）", "一般向け（基本的な説明）", "専門家向け（詳細な分析）"],
                                    index=1
                                )
                            
                            # プロンプト生成ボタン
                            generate_prompt = st.form_submit_button("📝 解釈プロンプトを生成", type="primary", use_container_width=True)
                        
                        if generate_prompt:
                            # 分析が実行されているか確認
                            if len(data_for_analysis) == 0:
                                st.error("先に分析を実行してください")
                            else:
                                # 分析結果をまとめる
                                analysis_summary = {
                                "データ概要": {
                                    "サンプル数": len(data_for_analysis),
                                    "ターゲット変数": target_col,
                                    "説明変数数": len(feature_cols),
                                    "説明変数リスト": feature_cols
                                },
                                "基本統計": {
                                    "ターゲット変数の平均": f"{y.mean():.2f}",
                                    "ターゲット変数の標準偏差": f"{y.std():.2f}",
                                    "ターゲット変数の最小値": f"{y.min():.2f}",
                                    "ターゲット変数の最大値": f"{y.max():.2f}"
                                }
                            }
                            
                            # 相関分析の結果
                            if 'corr_df' in locals():
                                top_correlations = []
                                for i, row in corr_df.head(5).iterrows():
                                    top_correlations.append({
                                        "変数": row['変数'],
                                        "相関係数": f"{row['相関係数']:.3f}",
                                        "p値": f"{row['p値']:.4f}",
                                        "有意性": row['有意性']
                                    })
                                analysis_summary["相関分析"] = {
                                    "最強相関変数": corr_df.iloc[0]['変数'],
                                    "最強相関係数": f"{corr_df.iloc[0]['相関係数']:.3f}",
                                    "上位5変数": top_correlations
                                }
                            
                            # 回帰分析の結果
                            if 'model' in locals() and 'model_sm' in locals():
                                analysis_summary["回帰分析"] = {
                                    "R2スコア": f"{test_r2:.4f}",
                                    "調整済みR2": f"{model_sm.rsquared_adj:.4f}",
                                    "RMSE": f"{test_rmse:.4f}",
                                    "F統計量": f"{model_sm.fvalue:.4f}",
                                    "F検定p値": f"{model_sm.f_pvalue:.6f}"
                                }
                                
                                # 重要な変数
                                if 'coef_df' in locals():
                                    top_coefficients = []
                                    for i, row in coef_df.head(5).iterrows():
                                        top_coefficients.append({
                                            "変数": row['変数'],
                                            "回帰係数": f"{row['回帰係数']:.4f}"
                                        })
                                    analysis_summary["回帰分析"]["影響度上位変数"] = top_coefficients
                            
                            # 決定木分析の結果
                            if 'importance_df' in locals():
                                top_importance = []
                                for i, row in importance_df.head(5).iterrows():
                                    top_importance.append({
                                        "変数": row['変数'],
                                        "重要度": f"{row['重要度']:.4f}"
                                    })
                                analysis_summary["決定木分析"] = {
                                    "R2スコア": f"{tree_r2:.4f}",
                                    "重要度上位変数": top_importance
                                }
                            
                            # シンプルな解釈用プロンプトテンプレート
                            if interpretation_level == "初心者向け（専門用語を避ける）":
                                level_instruction = "専門用語を使わず、誰にでも分かるように説明してください。"
                            elif interpretation_level == "専門家向け（詳細な分析）":
                                level_instruction = "統計的な詳細も含めて、専門的に解説してください。"
                            else:
                                level_instruction = "一般的なビジネスパーソンが理解できるレベルで説明してください。"
                            
                            # 分析データをJSON形式に整形
                            import json
                            analysis_data_str = json.dumps(analysis_summary, ensure_ascii=False, indent=2)
                            
                            # プロンプトテンプレート
                            interpretation_template = f"""
以下のデータ分析結果を解釈して、分かりやすく説明してください。

{level_instruction}

## 分析の背景
{business_context if business_context else "製品やサービスのシェアに影響する要因を分析しています。"}

## 分析結果
{analysis_data_str}

## 解釈してほしいポイント

1. **この分析結果から何が分かるか**
   - 最も重要な発見は何か
   - どの要因が最も影響力があるか
   
2. **なぜその結果になったのか**
   - 統計的に見て信頼できる結果か（R²値、p値の意味）
   - 変数間の関係性をどう理解すべきか
   
3. **実務への活用方法**
   - この結果をどう活用すればよいか
   - 優先的に取り組むべきことは何か
   
4. **注意すべき点**
   - この分析の限界は何か
   - 誤解しやすい点はあるか

分析結果を見て、上記のポイントについて解説をお願いします。
"""
                            
                            # プロンプトを生成
                            final_prompt = interpretation_template
                            
                            # プロンプト表示エリア
                            st.subheader("📄 生成されたプロンプト")
                            
                            # プロンプトを表示
                            prompt_container = st.container()
                            with prompt_container:
                                # テキストエリアに表示
                                prompt_area = st.text_area(
                                    "以下のプロンプトをコピーしてAIに貼り付けてください",
                                    value=final_prompt,
                                    height=400,
                                    key="generated_prompt"
                                )
                            
                            # コピーボタン（JavaScriptを使用）
                            col1, col2, col3 = st.columns([1, 1, 2])
                            with col1:
                                st.button("📋 クリップボードにコピー", key="copy_btn", help="プロンプトをクリップボードにコピーします")
                            with col2:
                                # 文字数表示
                                st.metric("文字数", f"{len(final_prompt):,}")
                            with col3:
                                st.info("💡 Ctrl+A → Ctrl+C でも全選択してコピーできます")
                            
                            # 使い方の説明
                            with st.expander("🔍 AIへの貼り付け方", expanded=False):
                                st.markdown("""
                                ### ChatGPT の場合
                                1. https://chat.openai.com にアクセス
                                2. 新しいチャットを開始
                                3. 生成されたプロンプトを貼り付けて送信
                                
                                ### Claude の場合
                                1. https://claude.ai にアクセス
                                2. 新しい会話を開始
                                3. プロンプトを貼り付けて送信
                                
                                ### Google Gemini の場合
                                1. https://gemini.google.com にアクセス
                                2. プロンプトを貼り付けて送信
                                
                                ### より良い結果を得るコツ
                                - 業界・分野を具体的に記入する
                                - ビジネス目標を明確にする
                                - 追加の背景情報があれば補足する
                                - AIからの回答に対して追加質問をする
                                """)
                            
                            # プロンプトのカスタマイズ
                            with st.expander("✏️ プロンプトのカスタマイズ", expanded=False):
                                st.markdown("### 追加の質問を含める")
                                
                                additional_questions = st.text_area(
                                    "AIに追加で聞きたいことがあれば記入してください",
                                    placeholder="例：\n- この業界特有の考慮事項は？\n- 中小企業でも実施可能な施策は？\n- デジタルマーケティングの観点では？",
                                    height=100
                                )
                                
                                if additional_questions and st.button("追加質問を含めて再生成"):
                                    final_prompt_with_additional = final_prompt + f"\n\n## 追加の質問\n{additional_questions}"
                                    st.text_area(
                                        "カスタマイズ済みプロンプト",
                                        value=final_prompt_with_additional,
                                        height=400,
                                        key="customized_prompt"
                                    )
                            
                            # プロンプトの保存
                            st.download_button(
                                label="💾 プロンプトをテキストファイルとして保存",
                                data=final_prompt,
                                file_name="ai_interpretation_prompt.txt",
                                mime="text/plain"
                            )
                    
                    # 完了メッセージ
                    status_text.text("分析完了！")
                    progress_bar.progress(100)
                    
                else:
                    st.warning("説明変数を少なくとも1つ選択してください。")
        else:
            st.warning("データに数値列が含まれていません。")
            
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        st.info("エンコーディング設定を変更して再度お試しください。")
        
else:
    # ファイルがアップロードされていない場合
    st.info("👈 左のサイドバーからCSVファイルをアップロードしてください。")
    
    # サンプルデータの説明
    with st.expander("📚 サンプルデータの形式"):
        st.markdown("""
        ### 期待されるデータ形式
        
        CSVファイルには以下のような列が含まれることを想定しています：
        
        - **ターゲット変数**（目的変数）: 製品シェア、売上シェアなど
        - **説明変数**（独立変数）: 
            - 価格
            - 広告費
            - 流通数
            - 顧客満足度
            - 競合数
            - 製品スペック評価
            - その他の数値データ
        
        ### データ例
        
        | 製品ID | シェア(%) | 価格 | 広告費 | 顧客満足度 | 流通数 |
        |--------|-----------|------|--------|------------|--------|
        | A001   | 25.5      | 1200 | 50000  | 4.2        | 150    |
        | A002   | 18.3      | 1500 | 30000  | 3.8        | 120    |
        | ...    | ...       | ...  | ...    | ...        | ...    |
        """)
    
    # 分析手法の説明
    with st.expander("🔬 分析手法の説明"):
        st.markdown("""
        ### 1. 相関分析
        - **Pearson相関係数**: 変数間の線形関係の強さを測定
        - **p値**: 統計的有意性を検定
        
        ### 2. 重回帰分析
        - 複数の説明変数がターゲット変数に与える影響を同時に分析
        - **R²スコア**: モデルの説明力を評価
        - **回帰係数**: 各変数の影響度を定量化
        
        ### 3. 主成分分析（PCA）
        - 多数の変数を少数の主成分に圧縮
        - データの次元削減と重要な特徴の抽出
        
        ### 4. 決定木分析
        - 変数の組み合わせによる分岐ルールを抽出
        - **変数重要度**: 予測における各変数の貢献度を評価
        """)