import streamlit as st
import pandas as pd
import numpy as np

# ページ設定
st.set_page_config(
    page_title="統計分析ガイド",
    page_icon="📚",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        margin: 15px 0;
    }
    .interpretation-box {
        background-color: #e8f5e9;
        padding: 15px;
        border-left: 4px solid #4caf50;
        border-radius: 5px;
        margin: 15px 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 15px;
        border-left: 4px solid #ff9800;
        border-radius: 5px;
        margin: 15px 0;
    }
    .example-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-left: 4px solid #2196f3;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 統計分析ガイド")
st.markdown("MDVデータ分析で使用する統計手法の解説と解釈方法")
st.markdown("---")

# タブを作成
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 相関分析", 
    "📉 回帰分析", 
    "🔍 多重共線性", 
    "📊 外れ値", 
    "🔄 交差検証", 
    "⚡ 正則化", 
    "🔬 回帰診断", 
    "💡 解釈のコツ"
])

# 1. 相関分析
with tab1:
    st.header("📈 相関分析")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("相関係数とは？")
        st.write("2つの変数間の直線的な関係の強さと方向を表す指標です。-1から+1の値を取ります。")
        
        st.markdown("""
        <div class="formula-box">
        r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² × Σ(yi - ȳ)²]
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("相関係数の種類")
        
        comparison_df = pd.DataFrame({
            '種類': ['Pearson相関係数', 'Spearman相関係数'],
            '特徴': ['線形関係の強さを測定', '順位相関（単調関係）を測定'],
            '前提条件': ['正規分布を仮定', '正規分布を仮定しない'],
            '外れ値への感度': ['影響を受けやすい', '影響を受けにくい']
        })
        st.dataframe(comparison_df, use_container_width=True)
    
    with col2:
        st.subheader("相関の強さの判定基準")
        
        interpretation_df = pd.DataFrame({
            '相関係数の絶対値': ['0.9 〜 1.0', '0.7 〜 0.9', '0.5 〜 0.7', '0.3 〜 0.5', '0.0 〜 0.3'],
            '関係の強さ': ['非常に強い', '強い', '中程度', '弱い', 'ほとんどなし'],
            '判定': ['🟢', '🟢', '🟡', '🟡', '🔴']
        })
        st.dataframe(interpretation_df, use_container_width=True)
    
    st.markdown("""
    <div class="example-box">
    <b>📝 実例</b><br>
    製品Aシェアと営業人員数の相関係数 = 0.72<br>
    → 強い正の相関があり、営業人員数が多いほど製品Aのシェアが高い傾向がある
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ 注意点</b><br>
    • 相関は因果関係を意味しない<br>
    • 第3の変数（交絡因子）の影響を考慮する必要がある<br>
    • 非線形関係は検出できない
    </div>
    """, unsafe_allow_html=True)

# 2. 回帰分析
with tab2:
    st.header("📉 回帰分析")
    
    st.subheader("回帰分析の基本")
    st.write("1つ以上の説明変数（X）から目的変数（Y）を予測する統計手法です。")
    
    st.markdown("""
    <div class="formula-box">
    Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ + ε<br>
    β₀: 切片、β₁〜βₙ: 回帰係数、ε: 誤差項
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("主要な評価指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R²（決定係数）", "モデルが説明できる変動の割合")
        st.info("""
        - 0.8以上: 優秀 🟢
        - 0.5〜0.8: 良好 🟡
        - 0.5未満: 改善必要 🔴
        """)
    
    with col2:
        st.metric("調整済みR²", "変数の数を考慮した決定係数")
        st.info("""
        - 変数を増やしても自動的に上がらない
        - モデル比較に適している
        """)
    
    with col3:
        st.metric("p値", "係数が0である確率")
        st.info("""
        - p < 0.001: *** 🟢
        - p < 0.01: ** 🟢
        - p < 0.05: * 🟡
        - p ≥ 0.05: 有意でない 🔴
        """)
    
    with col4:
        st.metric("RMSE", "予測誤差の標準偏差")
        st.info("""
        - 元のデータと同じ単位
        - 小さいほど良い
        """)
    
    st.markdown("""
    <div class="interpretation-box">
    <b>回帰係数の解釈</b><br>
    例: 営業人員数の係数 = 0.35<br>
    → 営業人員が1人増えると、製品Aシェアが0.35％増加する（他の変数が一定の場合）
    </div>
    """, unsafe_allow_html=True)

# 3. 多重共線性
with tab3:
    st.header("🔍 多重共線性（Multicollinearity）")
    
    st.subheader("VIF（分散拡大係数）")
    st.write("説明変数間の相関が高すぎる状態を検出する指標です。")
    
    st.markdown("""
    <div class="formula-box">
    VIF = 1 / (1 - R²ᵢ)<br>
    R²ᵢ: 変数iを他の変数で回帰した時の決定係数
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("VIFの判定基準")
    
    vif_df = pd.DataFrame({
        'VIF値': ['VIF < 5', '5 ≤ VIF < 10', 'VIF ≥ 10'],
        '判定': ['問題なし 🟢', '中程度の多重共線性 🟡', '深刻な多重共線性 🔴'],
        '対処法': ['そのまま使用可能', '注意が必要、変数の削除を検討', '変数の削除、主成分分析、Ridge回帰の使用']
    })
    st.dataframe(vif_df, use_container_width=True)
    
    st.markdown("""
    <div class="example-box">
    <b>📝 実例</b><br>
    売上高のVIF = 15.3<br>
    → 売上高は他の変数（従業員数など）と強く相関している<br>
    → どちらか一方を削除するか、主成分分析を検討
    </div>
    """, unsafe_allow_html=True)

# 4. 外れ値
with tab4:
    st.header("📊 外れ値の検出と処理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("IQR法（四分位範囲法）")
        st.write("データの中央50%の範囲を基準に外れ値を検出します。")
        
        st.markdown("""
        <div class="formula-box">
        下限 = Q1 - 1.5 × IQR<br>
        上限 = Q3 + 1.5 × IQR<br>
        (IQR = Q3 - Q1)
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Z-score法")
        st.write("平均からの標準偏差の倍数で外れ値を判定します。")
        
        st.markdown("""
        <div class="formula-box">
        Z = (x - μ) / σ<br>
        |Z| > 3 の場合、外れ値と判定
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("外れ値の処理方法")
    
    outlier_df = pd.DataFrame({
        '方法': ['キャッピング', '削除', '変換'],
        '説明': ['外れ値を境界値に置換', '外れ値を含む行を除外', '対数変換などで正規化'],
        '使用場面': ['データを保持したい場合', '明らかな異常値の場合', '分布が歪んでいる場合']
    })
    st.dataframe(outlier_df, use_container_width=True)
    
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ 注意点</b><br>
    • 外れ値が重要な情報を含む可能性がある<br>
    • ドメイン知識に基づいて判断する<br>
    • 処理前後で結果がどう変わるか確認する
    </div>
    """, unsafe_allow_html=True)

# 5. 交差検証
with tab5:
    st.header("🔄 交差検証（Cross-Validation）")
    
    st.subheader("K-fold交差検証")
    st.write("データをK個に分割し、K回の学習と検証を行う手法です。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("5-fold交差検証の流れ")
        st.markdown("""
        1. データを5つに分割
        2. 4つで学習、1つで検証を5回繰り返し
        3. 5回の結果を平均化
        """)
    
    with col2:
        st.subheader("結果の見方")
        st.markdown("""
        <div class="interpretation-box">
        <b>例: R² = 0.75 (±0.08)</b><br>
        • 平均R²: 0.75（モデルの性能）<br>
        • 標準偏差: 0.08（安定性）<br>
        • 標準偏差が小さい = 安定したモデル
        </div>
        """, unsafe_allow_html=True)
    
    st.info("""
    💡 **交差検証の利点**
    - 過学習の検出
    - モデルの汎化性能の評価
    - 限られたデータの有効活用
    """)

# 6. 正則化
with tab6:
    st.header("⚡ 正則化回帰")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Ridge回帰（L2正則化）")
        st.info("""
        **係数の2乗和にペナルティ**
        - 全変数を使用（係数を0にしない）
        - 多重共線性に強い
        - 係数を縮小
        """)
    
    with col2:
        st.subheader("Lasso回帰（L1正則化）")
        st.info("""
        **係数の絶対値和にペナルティ**
        - 変数選択（係数を0にする）
        - 解釈しやすいモデル
        - スパースな解
        """)
    
    with col3:
        st.subheader("Elastic Net")
        st.info("""
        **RidgeとLassoの組み合わせ**
        - 両方の利点を活用
        - グループ化された変数に強い
        - バランスの取れた手法
        """)
    
    st.subheader("α（正則化パラメータ）の選び方")
    st.write("""
    - **小さいα**: 通常の回帰に近い（過学習のリスク）
    - **大きいα**: 強い正則化（アンダーフィッティングのリスク）
    - **最適なα**: 交差検証で決定
    """)
    
    st.markdown("""
    <div class="example-box">
    <b>📝 Lassoで変数選択された例</b><br>
    元の変数: 30個 → 選択された変数: 8個<br>
    → 8個の重要な変数だけで予測可能<br>
    → シンプルで解釈しやすいモデル
    </div>
    """, unsafe_allow_html=True)

# 7. 回帰診断
with tab7:
    st.header("🔬 回帰診断")
    
    st.subheader("回帰の前提条件")
    st.markdown("""
    1. **線形性**: XとYの関係が線形
    2. **独立性**: 誤差項が独立
    3. **等分散性**: 誤差の分散が一定
    4. **正規性**: 誤差が正規分布
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("主要な検定")
        
        test_df = pd.DataFrame({
            '検定': ['Shapiro-Wilk検定', 'Breusch-Pagan検定', 'Durbin-Watson統計量', "Cook's Distance"],
            '目的': ['残差の正規性', '等分散性', '自己相関', '影響力のある観測値'],
            '良好な基準': ['p > 0.05', 'p > 0.05', '1.5〜2.5', '< 4/n']
        })
        st.dataframe(test_df, use_container_width=True)
    
    with col2:
        st.subheader("問題への対処法")
        
        solution_df = pd.DataFrame({
            '問題': ['非正規性', '不等分散', '自己相関', '影響力大の観測値'],
            '対処法': ['変数変換、ロバスト回帰', '加重最小二乗法', '時系列モデル', 'データ確認、外れ値処理']
        })
        st.dataframe(solution_df, use_container_width=True)

# 8. 解釈のコツ
with tab8:
    st.header("💡 分析結果の総合的な解釈")
    
    st.subheader("モデル選択の基準")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **AIC（赤池情報量基準）**
        - 予測精度重視
        - 小さいほど良いモデル
        """)
    
    with col2:
        st.info("""
        **BIC（ベイズ情報量基準）**
        - シンプルさ重視
        - 小さいほど良いモデル
        """)
    
    st.subheader("📊 総合的な判断例")
    
    st.markdown("""
    <div class="example-box">
    <b>分析結果サマリー</b><br>
    • 調整済みR² = 0.72 → <span style="color: green;">良好な説明力 ✓</span><br>
    • 交差検証R² = 0.68 (±0.05) → <span style="color: green;">安定した性能 ✓</span><br>
    • VIF最大値 = 4.2 → <span style="color: green;">多重共線性なし ✓</span><br>
    • Durbin-Watson = 1.95 → <span style="color: green;">自己相関なし ✓</span><br>
    • 正規性p値 = 0.03 → <span style="color: orange;">やや非正規 ⚠️</span><br>
    <br>
    <b>結論</b><br>
    モデルは概ね良好。正規性の問題は軽微なため、このまま使用可能。
    ただし、予測の信頼区間を解釈する際は注意が必要。
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("実務での活用ポイント")
    st.markdown("""
    1. 単一の指標だけでなく、複数の指標を総合的に判断
    2. 統計的有意性だけでなく、実務的な意味を考慮
    3. モデルの前提条件を満たしているか確認
    4. 予測の不確実性を認識し、適切な意思決定を行う
    """)
    
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ よくある誤解</b><br>
    <br>
    <b>R²が高い = 良いモデル</b><br>
    → 過学習の可能性。交差検証で確認必要<br>
    <br>
    <b>p値 < 0.05 = 重要な変数</b><br>
    → 効果の大きさも確認。実務的意味を考慮<br>
    <br>
    <b>相関が高い = 因果関係</b><br>
    → 相関≠因果。理論的背景の検討が必要<br>
    <br>
    <b>複雑なモデル = 良いモデル</b><br>
    → シンプルで解釈しやすいモデルが実務では有用
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("🎯 分析の流れ")
    
    with st.expander("推奨される分析ステップ", expanded=True):
        st.markdown("""
        1. **データの前処理**
           - 外れ値の検出と処理
           - 欠損値の確認
        
        2. **探索的データ分析**
           - 基本統計量の確認
           - 相関分析
        
        3. **モデル構築前の確認**
           - 多重共線性チェック（VIF）
           - 前提条件の検証
        
        4. **モデル構築と評価**
           - 複数手法の比較（線形、正則化、決定木）
           - 交差検証による性能評価
        
        5. **モデル診断**
           - 残差分析
           - 影響力のある観測値の確認
        
        6. **結果の解釈と報告**
           - 統計的意味と実務的意味の両面から解釈
           - 限界と改善点の明記
        """)

# フッター
st.markdown("---")
st.markdown("© 2024 MDVデータ分析ガイド | 統計手法の理解と実践")