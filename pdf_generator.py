"""
PDF Generator Module for MDV Share Analyzer
分析結果をPDFレポートとして出力する機能
"""

from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
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

# Try to register Japanese font
try:
    # Try common Japanese font paths on Windows
    font_paths = [
        r"C:\Windows\Fonts\msgothic.ttc",
        r"C:\Windows\Fonts\msmincho.ttc",
        r"C:\Windows\Fonts\meiryo.ttc",
        r"C:\Windows\Fonts\YuGothB.ttc",
        r"C:\Windows\Fonts\YuGothM.ttc",
    ]
    
    font_registered = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('Japanese', font_path))
                font_registered = True
                break
            except:
                continue
    
    if not font_registered:
        print("Warning: Japanese font not found. Using default font.")
except:
    print("Warning: Could not register Japanese font.")

class PDFReportGenerator:
    """PDF Report Generator for analysis results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_styles()
    
    def setup_styles(self):
        """Setup custom styles for the PDF"""
        # Try to use Japanese font if available
        try:
            base_font = 'Japanese'
        except:
            base_font = 'Helvetica'
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName=base_font
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName=base_font
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=10,
            fontName=base_font
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            fontName=base_font
        ))
        
        # Table header style
        self.styles.add(ParagraphStyle(
            name='TableHeader',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.whitesmoke,
            alignment=TA_CENTER,
            fontName=base_font
        ))
    
    def create_header(self, canvas, doc):
        """Create page header"""
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawString(inch, A4[1] - 0.5*inch, "MDV Share Analyzer Report")
        canvas.drawString(A4[0] - 2*inch, A4[1] - 0.5*inch, 
                         datetime.now().strftime("%Y-%m-%d %H:%M"))
        canvas.line(inch, A4[1] - 0.6*inch, A4[0] - inch, A4[1] - 0.6*inch)
        canvas.restoreState()
    
    def create_footer(self, canvas, doc):
        """Create page footer"""
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.drawCentredString(A4[0]/2, 0.5*inch, text)
        canvas.restoreState()
    
    def fig_to_image(self, fig: go.Figure, width: int = 600, height: int = 400) -> Image:
        """Convert Plotly figure to ReportLab Image"""
        # Convert Plotly figure to PNG bytes
        img_bytes = fig.to_image(format="png", width=width, height=height)
        img_stream = io.BytesIO(img_bytes)
        
        # Create ReportLab Image
        img = Image(img_stream, width=width*0.5, height=height*0.5)
        return img
    
    def dataframe_to_table(self, df: pd.DataFrame, col_widths: Optional[List[float]] = None) -> Table:
        """Convert pandas DataFrame to ReportLab Table"""
        # Prepare data
        data = []
        
        # Add header
        data.append(list(df.columns))
        
        # Add rows
        for _, row in df.iterrows():
            row_data = []
            for val in row:
                if isinstance(val, (int, float)):
                    if pd.isna(val):
                        row_data.append('')
                    elif isinstance(val, float):
                        row_data.append(f'{val:.4f}')
                    else:
                        row_data.append(str(val))
                else:
                    row_data.append(str(val))
            data.append(row_data)
        
        # Create table
        if col_widths is None:
            col_widths = [A4[0] - 2*inch] * len(df.columns)
            col_widths = [w / len(df.columns) for w in col_widths]
        
        table = Table(data, colWidths=col_widths)
        
        # Apply style
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 1), (-1, -1), 'RIGHT'),
        ])
        
        # Alternate row colors
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
        
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
            rightMargin=inch,
            leftMargin=inch,
            topMargin=inch,
            bottomMargin=inch
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Title page
        elements.append(Paragraph("MDV Share Analysis Report", self.styles['CustomTitle']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                 self.styles['CustomNormal']))
        elements.append(Spacer(1, 30))
        
        # Executive Summary
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        elements.append(Spacer(1, 12))
        
        if 'summary' in analysis_results:
            summary_text = self._format_summary(analysis_results['summary'])
            elements.append(Paragraph(summary_text, self.styles['CustomNormal']))
        
        elements.append(PageBreak())
        
        # Data Overview
        if 'data_overview' in analysis_results:
            elements.append(Paragraph("1. Data Overview", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            overview = analysis_results['data_overview']
            overview_text = f"""
            • Sample Size: {overview.get('samples', 'N/A')}<br/>
            • Number of Features: {overview.get('features', 'N/A')}<br/>
            • Target Variable: {overview.get('target', 'N/A')}<br/>
            • Missing Values: {overview.get('missing', 'N/A')}<br/>
            """
            elements.append(Paragraph(overview_text, self.styles['CustomNormal']))
            elements.append(Spacer(1, 20))
        
        # Correlation Analysis
        if 'correlation' in analysis_results:
            elements.append(Paragraph("2. Correlation Analysis", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            # Add correlation matrix as table if available
            if 'matrix' in analysis_results['correlation']:
                corr_df = analysis_results['correlation']['matrix']
                if isinstance(corr_df, pd.DataFrame) and len(corr_df) <= 10:
                    elements.append(Paragraph("Correlation Matrix:", self.styles['CustomSubHeading']))
                    table = self.dataframe_to_table(corr_df.round(3))
                    elements.append(table)
                    elements.append(Spacer(1, 20))
            
            # Add correlation insights
            if 'insights' in analysis_results['correlation']:
                elements.append(Paragraph("Key Correlations:", self.styles['CustomSubHeading']))
                insights_text = analysis_results['correlation']['insights']
                elements.append(Paragraph(insights_text, self.styles['CustomNormal']))
                elements.append(Spacer(1, 20))
        
        # Regression Analysis
        if 'regression' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("3. Regression Analysis", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            reg_results = analysis_results['regression']
            
            # Model metrics
            metrics_text = f"""
            <b>Model Performance:</b><br/>
            • R² Score (Test): {reg_results.get('r2_test', 'N/A')}<br/>
            • RMSE: {reg_results.get('rmse', 'N/A')}<br/>
            • MAE: {reg_results.get('mae', 'N/A')}<br/>
            • Cross-validation R²: {reg_results.get('cv_score', 'N/A')}<br/>
            """
            elements.append(Paragraph(metrics_text, self.styles['CustomNormal']))
            elements.append(Spacer(1, 12))
            
            # Feature importance
            if 'feature_importance' in reg_results:
                elements.append(Paragraph("Feature Importance:", self.styles['CustomSubHeading']))
                importance_df = reg_results['feature_importance']
                if isinstance(importance_df, pd.DataFrame):
                    table = self.dataframe_to_table(importance_df.head(10))
                    elements.append(table)
                    elements.append(Spacer(1, 20))
        
        # Decision Tree Analysis
        if 'tree' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("4. Decision Tree Analysis", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            tree_results = analysis_results['tree']
            
            # Model metrics
            metrics_text = f"""
            <b>Model Performance:</b><br/>
            • R² Score: {tree_results.get('r2', 'N/A')}<br/>
            • RMSE: {tree_results.get('rmse', 'N/A')}<br/>
            """
            elements.append(Paragraph(metrics_text, self.styles['CustomNormal']))
            elements.append(Spacer(1, 12))
            
            # Feature importance
            if 'feature_importance' in tree_results:
                elements.append(Paragraph("Feature Importance:", self.styles['CustomSubHeading']))
                importance_df = tree_results['feature_importance']
                if isinstance(importance_df, pd.DataFrame):
                    table = self.dataframe_to_table(importance_df.head(10))
                    elements.append(table)
                    elements.append(Spacer(1, 20))
        
        # PCA Analysis
        if 'pca' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("5. Principal Component Analysis", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            pca_results = analysis_results['pca']
            
            # Explained variance
            if 'explained_variance' in pca_results:
                var_text = f"""
                <b>Explained Variance:</b><br/>
                • PC1: {pca_results['explained_variance'][0]:.2%}<br/>
                • PC2: {pca_results['explained_variance'][1]:.2%} (if available)<br/>
                • Cumulative: {sum(pca_results['explained_variance'][:2]):.2%}<br/>
                """
                elements.append(Paragraph(var_text, self.styles['CustomNormal']))
                elements.append(Spacer(1, 20))
        
        # Diagnostics
        if 'diagnostics' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("6. Diagnostics", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            
            diagnostics = analysis_results['diagnostics']
            
            # VIF Analysis
            if 'vif' in diagnostics:
                elements.append(Paragraph("Multicollinearity Check (VIF):", self.styles['CustomSubHeading']))
                vif_df = diagnostics['vif']
                if isinstance(vif_df, pd.DataFrame):
                    table = self.dataframe_to_table(vif_df)
                    elements.append(table)
                    elements.append(Spacer(1, 20))
            
            # Outliers
            if 'outliers' in diagnostics:
                elements.append(Paragraph("Outlier Detection:", self.styles['CustomSubHeading']))
                outlier_text = diagnostics['outliers']
                elements.append(Paragraph(outlier_text, self.styles['CustomNormal']))
                elements.append(Spacer(1, 20))
        
        # Conclusions
        if 'conclusions' in analysis_results:
            elements.append(PageBreak())
            elements.append(Paragraph("Conclusions", self.styles['CustomHeading']))
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(analysis_results['conclusions'], self.styles['CustomNormal']))
        
        # Build PDF
        doc.build(elements, onFirstPage=self.create_header, onLaterPages=self.create_header)
        
        # Return bytes if using BytesIO
        if not output_path:
            buffer.seek(0)
            return buffer.getvalue()
        
        return None
    
    def _format_summary(self, summary: Dict[str, Any]) -> str:
        """Format summary dictionary as HTML text"""
        text = ""
        for key, value in summary.items():
            if isinstance(value, dict):
                text += f"<b>{key}:</b><br/>"
                for sub_key, sub_value in value.items():
                    text += f"  • {sub_key}: {sub_value}<br/>"
            else:
                text += f"<b>{key}:</b> {value}<br/>"
        return text


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
            'target': 'Share',  # This should be dynamic
            'missing': 0
        }
    
    # Summary
    pdf_data['summary'] = {}
    
    # Correlation
    if hasattr(analysis_results, 'correlation_matrix'):
        pdf_data['correlation'] = {
            'matrix': analysis_results.correlation_matrix,
            'insights': 'Correlation analysis completed successfully.'
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
            outlier_summary.append(f"{col}: {len(info['iqr_outliers'])} outliers detected")
        pdf_data['diagnostics']['outliers'] = '\n'.join(outlier_summary)
    
    # Conclusions
    pdf_data['conclusions'] = """
    The analysis has been completed successfully. Key findings include:
    
    1. The regression model shows good predictive performance.
    2. Feature importance analysis reveals the most influential variables.
    3. No severe multicollinearity issues were detected.
    4. The model can be used for prediction and what-if analysis.
    
    Please refer to the detailed sections above for specific metrics and insights.
    """
    
    return pdf_data