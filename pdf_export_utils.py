"""
PDF Export Utilities for MDV Share Analyzer
グラフと表を個別にエクスポートできる機能
"""

import io
import base64
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from typing import Dict, Any, List, Optional
import streamlit as st

def export_chart_as_image(fig: go.Figure, format: str = 'png', width: int = 1200, height: int = 800) -> bytes:
    """
    Export Plotly figure as image bytes
    """
    try:
        if format == 'png':
            img_bytes = fig.to_image(format="png", width=width, height=height)
        elif format == 'pdf':
            img_bytes = fig.to_image(format="pdf", width=width, height=height)
        elif format == 'svg':
            img_bytes = fig.to_image(format="svg", width=width, height=height)
        else:
            img_bytes = fig.to_image(format="png", width=width, height=height)
        return img_bytes
    except Exception as e:
        st.error(f"画像エクスポートエラー: {str(e)}")
        return None

def export_table_as_csv(df: pd.DataFrame, encoding: str = 'utf-8-sig') -> bytes:
    """
    Export DataFrame as CSV bytes
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding=encoding)
    return csv_buffer.getvalue().encode(encoding)

def export_table_as_excel(dfs: Dict[str, pd.DataFrame]) -> bytes:
    """
    Export multiple DataFrames as Excel file with multiple sheets
    """
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet name limit
    excel_buffer.seek(0)
    return excel_buffer.getvalue()

def create_chart_collection_pdf(charts: Dict[str, go.Figure], title: str = "Chart Collection") -> bytes:
    """
    Create PDF with multiple charts
    """
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, PageBreak, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    
    # Register Japanese fonts
    try:
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
        font_name = 'HeiseiKakuGo-W5'
    except:
        font_name = 'Helvetica'
    
    # Create PDF buffer
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Setup styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1,  # Center
        fontName=font_name
    )
    
    chart_title_style = ParagraphStyle(
        'ChartTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName=font_name
    )
    
    # Build content
    elements = []
    
    # Title page
    elements.append(Paragraph(title, title_style))
    elements.append(Paragraph(f"作成日: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}", styles['Normal']))
    elements.append(PageBreak())
    
    # Add each chart
    for chart_name, fig in charts.items():
        # Chart title
        elements.append(Paragraph(chart_name, chart_title_style))
        
        # Convert chart to image
        try:
            img_bytes = fig.to_image(format="png", width=1000, height=600)
            img_buffer = io.BytesIO(img_bytes)
            img = Image(img_buffer, width=9*inch, height=5.4*inch)
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"グラフ生成エラー: {str(e)}", styles['Normal']))
        
        elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def create_table_collection_pdf(tables: Dict[str, pd.DataFrame], title: str = "Table Collection") -> bytes:
    """
    Create PDF with multiple tables
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, PageBreak, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    
    # Register Japanese fonts
    try:
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
        font_name = 'HeiseiKakuGo-W5'
        font_name_min = 'HeiseiMin-W3'
    except:
        font_name = 'Helvetica'
        font_name_min = 'Helvetica'
    
    # Create PDF buffer
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=inch,
        bottomMargin=inch
    )
    
    # Setup styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1,  # Center
        fontName=font_name
    )
    
    table_title_style = ParagraphStyle(
        'TableTitle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName=font_name
    )
    
    cell_style = ParagraphStyle(
        'CellStyle',
        parent=styles['Normal'],
        fontSize=9,
        fontName=font_name_min
    )
    
    # Build content
    elements = []
    
    # Title page
    elements.append(Paragraph(title, title_style))
    elements.append(Paragraph(f"作成日: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))
    
    # Table of contents
    elements.append(Paragraph("目次", table_title_style))
    for i, table_name in enumerate(tables.keys(), 1):
        elements.append(Paragraph(f"{i}. {table_name}", styles['Normal']))
    elements.append(PageBreak())
    
    # Add each table
    for table_name, df in tables.items():
        # Table title
        elements.append(Paragraph(table_name, table_title_style))
        
        # Prepare table data
        data = []
        
        # Header
        header_row = []
        for col in df.columns:
            header_row.append(Paragraph(str(col), cell_style))
        data.append(header_row)
        
        # Data rows (limit to 50 rows for PDF)
        for idx, row in df.head(50).iterrows():
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
                    text = str(val)
                row_data.append(Paragraph(text, cell_style))
            data.append(row_data)
        
        # Create table
        col_widths = [(A4[0] - 1.5*inch) / len(df.columns)] * len(df.columns)
        table = Table(data, colWidths=col_widths, repeatRows=1)
        
        # Apply style
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ])
        table.setStyle(table_style)
        
        elements.append(table)
        
        # Add note if table was truncated
        if len(df) > 50:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph(f"※ 表示は最初の50行のみ（全{len(df)}行）", styles['Normal']))
        
        elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def create_export_ui(analysis_results, target_col: str = None):
    """
    Create UI for exporting charts and tables
    """
    st.header("📊 グラフ・表エクスポート")
    
    # Export type selection
    export_type = st.radio(
        "エクスポートタイプを選択",
        ["グラフのみ", "表のみ", "両方", "個別選択"],
        horizontal=True
    )
    
    # Prepare available items
    available_charts = {}
    available_tables = {}
    
    # Correlation heatmap
    if hasattr(analysis_results, 'correlation_matrix'):
        corr_matrix = analysis_results.correlation_matrix
        fig_corr = go.Figure(data=go.Heatmap(
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
        fig_corr.update_layout(title="相関行列ヒートマップ", height=600)
        available_charts["相関行列ヒートマップ"] = fig_corr
        available_tables["相関行列"] = corr_matrix
    
    # Regression plots
    if hasattr(analysis_results, 'regression_results'):
        reg = analysis_results.regression_results
        
        # Actual vs Predicted
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=reg['y_test'],
            y=reg['y_pred'],
            mode='markers',
            name='予測値',
            marker=dict(size=8, opacity=0.6)
        ))
        min_val = min(reg['y_test'].min(), reg['y_pred'].min())
        max_val = max(reg['y_test'].max(), reg['y_pred'].max())
        fig_reg.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='理想線',
            line=dict(dash='dash', color='red')
        ))
        fig_reg.update_layout(
            title=f"予測値 vs 実測値 (R²={reg['test_r2']:.4f})",
            xaxis_title="実測値",
            yaxis_title="予測値",
            height=600
        )
        available_charts["回帰分析: 予測vs実測"] = fig_reg
        
        # Feature importance
        if reg['feature_importance'] is not None:
            importance_df = reg['feature_importance'].head(10)
            fig_importance = go.Figure(go.Bar(
                x=importance_df['Abs_Coefficient'] if 'Abs_Coefficient' in importance_df.columns else importance_df['Coefficient'],
                y=importance_df['Variable'],
                orientation='h'
            ))
            fig_importance.update_layout(
                title="回帰係数（上位10変数）",
                xaxis_title="係数の絶対値",
                yaxis_title="変数",
                height=600
            )
            available_charts["回帰分析: 特徴量重要度"] = fig_importance
            available_tables["回帰係数"] = importance_df
    
    # Decision tree
    if hasattr(analysis_results, 'tree_results'):
        tree = analysis_results.tree_results
        if 'feature_importance' in tree:
            importance_df = tree['feature_importance'].head(10)
            fig_tree = go.Figure(go.Bar(
                x=importance_df['Importance'],
                y=importance_df['Variable'],
                orientation='h'
            ))
            fig_tree.update_layout(
                title="決定木: 特徴量重要度（上位10変数）",
                xaxis_title="重要度",
                yaxis_title="変数",
                height=600
            )
            available_charts["決定木: 特徴量重要度"] = fig_tree
            available_tables["決定木特徴量重要度"] = importance_df
    
    # PCA
    if hasattr(analysis_results, 'pca_results'):
        pca = analysis_results.pca_results
        
        # Scree plot
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(pca['explained_variance_ratio']))],
            y=pca['explained_variance_ratio'],
            name='寄与率'
        ))
        fig_scree.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(pca['cumulative_variance_ratio']))],
            y=pca['cumulative_variance_ratio'],
            mode='lines+markers',
            name='累積寄与率',
            yaxis='y2'
        ))
        fig_scree.update_layout(
            title="主成分の寄与率",
            yaxis=dict(title='寄与率'),
            yaxis2=dict(title='累積寄与率', overlaying='y', side='right'),
            height=600
        )
        available_charts["PCA: 寄与率"] = fig_scree
        
        # Loadings table
        available_tables["PCA負荷量"] = pca['loadings']
    
    # VIF table
    if hasattr(analysis_results, 'vif_results') and analysis_results.vif_results is not None:
        available_tables["VIF分析"] = analysis_results.vif_results
    
    # Selection UI based on export type
    selected_charts = []
    selected_tables = []
    
    if export_type == "個別選択":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("グラフ選択")
            for chart_name in available_charts.keys():
                if st.checkbox(chart_name, key=f"chart_{chart_name}"):
                    selected_charts.append(chart_name)
        
        with col2:
            st.subheader("表選択")
            for table_name in available_tables.keys():
                if st.checkbox(table_name, key=f"table_{table_name}"):
                    selected_tables.append(table_name)
    
    elif export_type == "グラフのみ":
        selected_charts = list(available_charts.keys())
    
    elif export_type == "表のみ":
        selected_tables = list(available_tables.keys())
    
    else:  # 両方
        selected_charts = list(available_charts.keys())
        selected_tables = list(available_tables.keys())
    
    # Export format selection
    st.subheader("エクスポート形式")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if selected_charts:
            st.write("**グラフのエクスポート形式:**")
            chart_format = st.selectbox(
                "形式",
                ["PDF (複数グラフ)", "PNG (個別)", "SVG (個別)"],
                key="chart_format"
            )
    
    with col2:
        if selected_tables:
            st.write("**表のエクスポート形式:**")
            table_format = st.selectbox(
                "形式",
                ["PDF (複数表)", "Excel (複数シート)", "CSV (個別)"],
                key="table_format"
            )
    
    # Export buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    # Export charts
    if selected_charts:
        with col1:
            if st.button("📊 グラフをエクスポート", type="primary", use_container_width=True):
                try:
                    if chart_format == "PDF (複数グラフ)":
                        # Create PDF with all selected charts
                        charts_to_export = {name: available_charts[name] for name in selected_charts}
                        pdf_bytes = create_chart_collection_pdf(charts_to_export, "MDV分析グラフ集")
                        st.download_button(
                            label="📥 グラフPDFをダウンロード",
                            data=pdf_bytes,
                            file_name=f"mdv_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    
                    elif chart_format == "PNG (個別)":
                        # Export each chart as PNG
                        for chart_name in selected_charts:
                            fig = available_charts[chart_name]
                            img_bytes = export_chart_as_image(fig, 'png')
                            if img_bytes:
                                st.download_button(
                                    label=f"📥 {chart_name}.png",
                                    data=img_bytes,
                                    file_name=f"{chart_name.replace(':', '_')}_{datetime.now().strftime('%Y%m%d')}.png",
                                    mime="image/png"
                                )
                    
                    elif chart_format == "SVG (個別)":
                        # Export each chart as SVG
                        for chart_name in selected_charts:
                            fig = available_charts[chart_name]
                            img_bytes = export_chart_as_image(fig, 'svg')
                            if img_bytes:
                                st.download_button(
                                    label=f"📥 {chart_name}.svg",
                                    data=img_bytes,
                                    file_name=f"{chart_name.replace(':', '_')}_{datetime.now().strftime('%Y%m%d')}.svg",
                                    mime="image/svg+xml"
                                )
                    
                    st.success(f"✅ {len(selected_charts)}個のグラフをエクスポートしました")
                
                except Exception as e:
                    st.error(f"グラフエクスポートエラー: {str(e)}")
    
    # Export tables
    if selected_tables:
        with col2:
            if st.button("📋 表をエクスポート", type="primary", use_container_width=True):
                try:
                    if table_format == "PDF (複数表)":
                        # Create PDF with all selected tables
                        tables_to_export = {name: available_tables[name] for name in selected_tables}
                        pdf_bytes = create_table_collection_pdf(tables_to_export, "MDV分析データ表")
                        st.download_button(
                            label="📥 表PDFをダウンロード",
                            data=pdf_bytes,
                            file_name=f"mdv_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                    
                    elif table_format == "Excel (複数シート)":
                        # Export all tables as Excel with multiple sheets
                        tables_to_export = {name: available_tables[name] for name in selected_tables}
                        excel_bytes = export_table_as_excel(tables_to_export)
                        st.download_button(
                            label="📥 Excelファイルをダウンロード",
                            data=excel_bytes,
                            file_name=f"mdv_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    elif table_format == "CSV (個別)":
                        # Export each table as CSV
                        for table_name in selected_tables:
                            df = available_tables[table_name]
                            csv_bytes = export_table_as_csv(df)
                            st.download_button(
                                label=f"📥 {table_name}.csv",
                                data=csv_bytes,
                                file_name=f"{table_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                    
                    st.success(f"✅ {len(selected_tables)}個の表をエクスポートしました")
                
                except Exception as e:
                    st.error(f"表エクスポートエラー: {str(e)}")
    
    # Combined export
    if selected_charts and selected_tables:
        with col3:
            if st.button("📦 すべてエクスポート", type="secondary", use_container_width=True):
                try:
                    # Create ZIP file with all exports
                    import zipfile
                    
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Add charts PDF
                        charts_to_export = {name: available_charts[name] for name in selected_charts}
                        charts_pdf = create_chart_collection_pdf(charts_to_export, "MDV分析グラフ集")
                        zipf.writestr("charts.pdf", charts_pdf)
                        
                        # Add tables Excel
                        tables_to_export = {name: available_tables[name] for name in selected_tables}
                        excel_bytes = export_table_as_excel(tables_to_export)
                        zipf.writestr("tables.xlsx", excel_bytes)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label="📥 ZIPファイルをダウンロード",
                        data=zip_buffer.getvalue(),
                        file_name=f"mdv_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
                    
                    st.success("✅ すべてのデータをZIPファイルにエクスポートしました")
                
                except Exception as e:
                    st.error(f"エクスポートエラー: {str(e)}")