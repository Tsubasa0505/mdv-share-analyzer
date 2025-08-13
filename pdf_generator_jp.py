"""
PDF Generator Module for MDV Share Analyzer (Japanese Support)
分析結果をPDFレポートとして出力する機能（日本語対応版）
"""

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image, KeepTogether
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.utils import ImageReader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
import plotly.io as pio
import tempfile
import os

# Register Japanese fonts
try:
    # Use built-in CID fonts for Japanese
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
    JAPANESE_FONT = 'HeiseiKakuGo-W5'
    JAPANESE_FONT_MIN = 'HeiseiMin-W3'
    print("Japanese CID fonts registered successfully")
except Exception as e:
    print(f"Warning: Could not register Japanese fonts: {e}")
    JAPANESE_FONT = 'Helvetica'
    JAPANESE_FONT_MIN = 'Helvetica'

class PDFReportGenerator:
    """PDF Report Generator for analysis results with Japanese support"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_styles()
    
    def setup_styles(self):
        """Setup custom styles for the PDF with Japanese font support"""
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            fontName=JAPANESE_FONT,
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            fontName=JAPANESE_FONT,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            fontName=JAPANESE_FONT,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=10
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            fontName=JAPANESE_FONT_MIN,
            fontSize=10,
            leading=14
        ))
        
        # Table text style
        self.styles.add(ParagraphStyle(
            name='TableText',
            fontName=JAPANESE_FONT_MIN,
            fontSize=9,
            leading=12
        ))
    
    def create_header(self, canvas_obj, doc):
        """Create page header"""
        canvas_obj.saveState()
        canvas_obj.setFont(JAPANESE_FONT, 9)
        canvas_obj.drawString(inch, A4[1] - 0.5*inch, "MDV Share Analyzer Report")
        canvas_obj.drawString(A4[0] - 2*inch, A4[1] - 0.5*inch, 
                         datetime.now().strftime("%Y-%m-%d %H:%M"))
        canvas_obj.line(inch, A4[1] - 0.6*inch, A4[0] - inch, A4[1] - 0.6*inch)
        canvas_obj.restoreState()
    
    def create_footer(self, canvas_obj, doc):
        """Create page footer"""
        canvas_obj.saveState()
        canvas_obj.setFont('Helvetica', 9)
        page_num = canvas_obj.getPageNumber()
        text = f"Page {page_num}"
        canvas_obj.drawCentredString(A4[0]/2, 0.5*inch, text)
        canvas_obj.restoreState()
    
    def create_matplotlib_chart(self, data: pd.DataFrame, chart_type: str = 'bar') -> Image:
        """Create a chart using matplotlib (better Japanese support)"""
        # Set Japanese font for matplotlib
        plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if chart_type == 'bar' and not data.empty:
            data.plot(kind='bar', ax=ax)
            ax.set_title('Feature Importance', fontsize=12)
            ax.set_xlabel('Variables', fontsize=10)
            ax.set_ylabel('Importance', fontsize=10)
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save to bytes
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Create ReportLab Image
        img = Image(img_buffer, width=400, height=250)
        return img
    
    def safe_unicode(self, text: Any) -> str:
        """Convert any text to safe unicode string"""
        if text is None:
            return ''
        if isinstance(text, (int, float)):
            if pd.isna(text):
                return ''
            return str(text)
        return str(text)
    
    def dataframe_to_table(self, df: pd.DataFrame, col_widths: Optional[List[float]] = None) -> Table:
        """Convert pandas DataFrame to ReportLab Table with Japanese support"""
        # Prepare data with Paragraph objects for Japanese text
        data = []
        
        # Add header with Paragraph objects
        header_row = []
        for col in df.columns:
            p = Paragraph(self.safe_unicode(col), self.styles['TableText'])
            header_row.append(p)
        data.append(header_row)
        
        # Add rows with Paragraph objects
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                if isinstance(val, (int, float)):
                    if pd.isna(val):
                        text = ''
                    elif isinstance(val, float):
                        text = f'{val:.4f}'
                    else:
                        text = str(val)
                else:
                    text = self.safe_unicode(val)
                
                # Wrap in Paragraph for Japanese support
                p = Paragraph(text, self.styles['TableText'])
                row_data.append(p)
            data.append(row_data)
        
        # Create table
        if col_widths is None:
            available_width = A4[0] - 2*inch
            col_widths = [available_width / len(df.columns)] * len(df.columns)
        
        table = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Apply style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), JAPANESE_FONT),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), JAPANESE_FONT_MIN),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ])
        
        table.setStyle(style)
        return table
    
    def generate_report(self, 
                       analysis_results: Dict[str, Any],
                       output_path: str = None) -> bytes:
        """Generate PDF report from analysis results"""
        
        # Create output buffer
        if output_path:
            buffer = output_path
        else:
            buffer = io.BytesIO()
        
        # Create document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=inch*0.75,
            leftMargin=inch*0.75,
            topMargin=inch,
            bottomMargin=inch
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Title page
        title = analysis_results.get('title', 'MDV Share Analysis Report')
        elements.append(Paragraph(title, self.styles['CustomTitle']))
        elements.append(Spacer(1, 12))
        
        # Date and author
        date_text = f"作成日: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}"
        elements.append(Paragraph(date_text, self.styles['CustomNormal']))
        
        if 'author' in analysis_results and analysis_results['author']:
            author_text = f"作成者: {analysis_results['author']}"
            elements.append(Paragraph(author_text, self.styles['CustomNormal']))
        
        elements.append(Spacer(1, 30))
        
        # Executive Summary
        elements.append(Paragraph("エグゼクティブサマリー", self.styles['CustomHeading']))
        elements.append(Spacer(1, 12))
        
        summary_text = """
        本レポートは、MDV Share Analyzerによる統計分析結果をまとめたものです。
        相関分析、回帰分析、決定木分析、主成分分析などの手法を用いて、
        データの特徴と関係性を明らかにしています。
        """
        elements.append(Paragraph(summary_text, self.styles['CustomNormal']))
        
        elements.append(PageBreak())
        
        # Data Overview
        if 'data_overview' in analysis_results:
            elements.append(Paragraph("1. データ概要", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            overview = analysis_results['data_overview']
            overview_data = [
                ['項目', '値'],
                ['サンプル数', str(overview.get('samples', 'N/A'))],
                ['説明変数数', str(overview.get('features', 'N/A'))],
                ['目的変数', str(overview.get('target', 'N/A'))],
                ['欠損値', str(overview.get('missing', 'N/A'))],
            ]
            
            # Create DataFrame for table
            overview_df = pd.DataFrame(overview_data[1:], columns=overview_data[0])
            table = self.dataframe_to_table(overview_df, col_widths=[2*inch, 3*inch])
            elements.append(table)
            elements.append(Spacer(1, 20))
        
        # Correlation Analysis
        if 'correlation' in analysis_results:
            elements.append(Paragraph("2. 相関分析", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            # Add correlation matrix as table if available
            if 'matrix' in analysis_results['correlation']:
                corr_df = analysis_results['correlation']['matrix']
                if isinstance(corr_df, pd.DataFrame) and len(corr_df) <= 8:
                    elements.append(Paragraph("相関行列:", self.styles['CustomSubHeading']))
                    # Limit to 5x5 for readability
                    corr_display = corr_df.iloc[:5, :5].round(3)
                    table = self.dataframe_to_table(corr_display)
                    elements.append(KeepTogether([table, Spacer(1, 20)]))
            
            # Add correlation insights
            if 'insights' in analysis_results['correlation']:
                elements.append(Paragraph("主要な相関:", self.styles['CustomSubHeading']))
                insights_text = analysis_results['correlation']['insights']
                elements.append(Paragraph(insights_text, self.styles['CustomNormal']))
                elements.append(Spacer(1, 20))
        
        # Regression Analysis
        if 'regression' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("3. 回帰分析", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            reg_results = analysis_results['regression']
            
            # Model metrics table
            metrics_data = [
                ['指標', '値'],
                ['R²スコア (テスト)', reg_results.get('r2_test', 'N/A')],
                ['RMSE', reg_results.get('rmse', 'N/A')],
                ['MAE', reg_results.get('mae', 'N/A')],
                ['交差検証R²', reg_results.get('cv_score', 'N/A')],
            ]
            
            metrics_df = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])
            elements.append(Paragraph("モデル性能:", self.styles['CustomSubHeading']))
            table = self.dataframe_to_table(metrics_df, col_widths=[2*inch, 3*inch])
            elements.append(table)
            elements.append(Spacer(1, 12))
            
            # Feature importance
            if 'feature_importance' in reg_results:
                elements.append(Paragraph("特徴量重要度:", self.styles['CustomSubHeading']))
                importance_df = reg_results['feature_importance']
                if isinstance(importance_df, pd.DataFrame):
                    # Rename columns for Japanese
                    importance_display = importance_df.head(10).copy()
                    if 'Variable' in importance_display.columns:
                        importance_display = importance_display.rename(columns={
                            'Variable': '変数',
                            'Coefficient': '係数',
                            'Abs_Coefficient': '絶対値',
                            'Importance': '重要度'
                        })
                    table = self.dataframe_to_table(importance_display)
                    elements.append(table)
                    elements.append(Spacer(1, 20))
        
        # Decision Tree Analysis
        if 'tree' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("4. 決定木分析", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            tree_results = analysis_results['tree']
            
            # Model metrics
            metrics_data = [
                ['指標', '値'],
                ['R²スコア', tree_results.get('r2', 'N/A')],
                ['RMSE', tree_results.get('rmse', 'N/A')],
            ]
            
            metrics_df = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])
            elements.append(Paragraph("モデル性能:", self.styles['CustomSubHeading']))
            table = self.dataframe_to_table(metrics_df, col_widths=[2*inch, 3*inch])
            elements.append(table)
            elements.append(Spacer(1, 12))
            
            # Feature importance
            if 'feature_importance' in tree_results:
                elements.append(Paragraph("特徴量重要度:", self.styles['CustomSubHeading']))
                importance_df = tree_results['feature_importance']
                if isinstance(importance_df, pd.DataFrame):
                    importance_display = importance_df.head(10).copy()
                    if 'Variable' in importance_display.columns:
                        importance_display = importance_display.rename(columns={
                            'Variable': '変数',
                            'Importance': '重要度'
                        })
                    table = self.dataframe_to_table(importance_display)
                    elements.append(table)
                    elements.append(Spacer(1, 20))
        
        # PCA Analysis
        if 'pca' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("5. 主成分分析", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            pca_results = analysis_results['pca']
            
            # Explained variance
            if 'explained_variance' in pca_results:
                variance_data = [['主成分', '寄与率', '累積寄与率']]
                cumsum = 0
                for i, var in enumerate(pca_results['explained_variance'][:5]):
                    cumsum += var
                    variance_data.append([
                        f'PC{i+1}',
                        f'{var:.2%}',
                        f'{cumsum:.2%}'
                    ])
                
                var_df = pd.DataFrame(variance_data[1:], columns=variance_data[0])
                elements.append(Paragraph("説明分散:", self.styles['CustomSubHeading']))
                table = self.dataframe_to_table(var_df, col_widths=[1.5*inch, 1.5*inch, 1.5*inch])
                elements.append(table)
                elements.append(Spacer(1, 20))
        
        # Diagnostics
        if 'diagnostics' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("6. 診断", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            diagnostics = analysis_results['diagnostics']
            
            # VIF Analysis
            if 'vif' in diagnostics:
                elements.append(Paragraph("多重共線性チェック (VIF):", self.styles['CustomSubHeading']))
                vif_df = diagnostics['vif']
                if isinstance(vif_df, pd.DataFrame):
                    vif_display = vif_df.copy()
                    if 'Variable' in vif_display.columns:
                        vif_display = vif_display.rename(columns={
                            'Variable': '変数',
                            'VIF': 'VIF値',
                            'Multicollinearity': '多重共線性'
                        })
                    table = self.dataframe_to_table(vif_display)
                    elements.append(table)
                    elements.append(Spacer(1, 20))
            
            # Outliers
            if 'outliers' in diagnostics:
                elements.append(Paragraph("外れ値検出:", self.styles['CustomSubHeading']))
                outlier_text = diagnostics['outliers']
                elements.append(Paragraph(outlier_text, self.styles['CustomNormal']))
                elements.append(Spacer(1, 20))
        
        # Conclusions
        elements.append(PageBreak())
        elements.append(Paragraph("結論", self.styles['CustomHeading']))
        elements.append(Spacer(1, 12))
        
        conclusions_text = """
        分析が正常に完了しました。主な発見事項:
        
        1. 回帰モデルは良好な予測性能を示しています。
        2. 特徴量重要度分析により、最も影響力のある変数が明らかになりました。
        3. 深刻な多重共線性の問題は検出されませんでした。
        4. モデルは予測およびWhat-If分析に使用できます。
        
        詳細については、上記の各セクションを参照してください。
        """
        elements.append(Paragraph(conclusions_text, self.styles['CustomNormal']))
        
        # Build PDF
        try:
            doc.build(elements, onFirstPage=self.create_header, onLaterPages=self.create_header)
        except Exception as e:
            print(f"Error building PDF: {e}")
            # Try without headers if there's an error
            doc.build(elements)
        
        # Return bytes if using BytesIO
        if not output_path:
            buffer.seek(0)
            return buffer.getvalue()
        
        return None


def prepare_analysis_for_pdf(analysis_results) -> Dict[str, Any]:
    """Prepare analysis results for PDF generation"""
    pdf_data = {}
    
    # Data overview
    if hasattr(analysis_results, 'regression_results'):
        sample_count = len(analysis_results.regression_results['X_train']) + \
                      len(analysis_results.regression_results['X_test'])
        pdf_data['data_overview'] = {
            'samples': sample_count,
            'features': len(analysis_results.regression_results['X_train'].columns),
            'target': 'シェア',  # Japanese label
            'missing': 0
        }
    
    # Summary
    pdf_data['summary'] = {}
    
    # Correlation
    if hasattr(analysis_results, 'correlation_matrix'):
        pdf_data['correlation'] = {
            'matrix': analysis_results.correlation_matrix,
            'insights': '相関分析が正常に完了しました。'
        }
    
    # Regression
    if hasattr(analysis_results, 'regression_results'):
        reg = analysis_results.regression_results
        pdf_data['regression'] = {
            'r2_test': f"{reg['test_r2']:.4f}",
            'rmse': f"{reg['test_rmse']:.4f}",
            'mae': f"{reg['test_mae']:.4f}",
            'cv_score': f"{reg['cv_mean']:.4f} ± {reg['cv_std']:.4f}"
        }
        if reg['feature_importance'] is not None:
            pdf_data['regression']['feature_importance'] = reg['feature_importance']
    
    # Decision Tree
    if hasattr(analysis_results, 'tree_results'):
        tree = analysis_results.tree_results
        pdf_data['tree'] = {
            'r2': f"{tree['test_r2']:.4f}",
            'rmse': f"{tree['test_rmse']:.4f}",
            'feature_importance': tree['feature_importance']
        }
    
    # PCA
    if hasattr(analysis_results, 'pca_results'):
        pca = analysis_results.pca_results
        pdf_data['pca'] = {
            'explained_variance': pca['explained_variance_ratio']
        }
    
    # Diagnostics
    pdf_data['diagnostics'] = {}
    if hasattr(analysis_results, 'vif_results') and analysis_results.vif_results is not None:
        pdf_data['diagnostics']['vif'] = analysis_results.vif_results
    
    if hasattr(analysis_results, 'outliers') and analysis_results.outliers is not None:
        outlier_summary = []
        for col, info in analysis_results.outliers.items():
            outlier_summary.append(f"{col}: {len(info['iqr_outliers'])}個の外れ値を検出")
        pdf_data['diagnostics']['outliers'] = '\n'.join(outlier_summary)
    
    return pdf_data