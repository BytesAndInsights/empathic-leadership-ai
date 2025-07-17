# src/report_generator.py
"""
Report Generation Module
Creates PDF reports and visualizations for leadership assessments
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import tempfile
import base64
from pathlib import Path

class ReportGenerator:
    """Generate comprehensive leadership assessment reports"""
    
    def __init__(self):
        self.template_path = Path("templates")
        self.output_path = Path("reports")
        self.output_path.mkdir(exist_ok=True)
    
    def generate_full_report(self, assessment_results: Dict) -> str:
        """Generate complete PDF report"""
        
        # Create PDF
        pdf = CustomPDF()
        pdf.add_page()
        
        # Title page
        pdf.set_font("Arial", "B", 24)
        pdf.cell(0, 20, "Leadership Assessment Report", ln=True, align="C")
        
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%B %d, %Y')}", ln=True, align="C")
        pdf.ln(20)
        
        # Executive Summary
        pdf.add_section("Executive Summary")
        pdf.add_paragraph(self._generate_executive_summary(assessment_results))
        
        # Scores Overview
        pdf.add_section("Assessment Scores")
        pdf.add_paragraph(f"Traditional Assessment: {assessment_results.get('traditional_score', 0):.1f}/10")
        pdf.add_paragraph(f"Inclusive Assessment: {assessment_results.get('inclusive_score', 0):.1f}/10")
        pdf.add_paragraph(f"AI-Era Readiness: {self._calculate_ai_readiness(assessment_results):.0f}%")
        
        # Leadership Style
        pdf.add_section("Leadership Style Profile")
        for style, score in assessment_results.get("leadership_scores", {}).items():
            pdf.add_paragraph(f"• {style.replace('_', ' ').title()}: {score:.1f}/10")
        
        # Competencies
        pdf.add_section("Core Competencies")
        competencies = assessment_results.get("competencies", {})
        for comp, score in sorted(competencies.items(), key=lambda x: x[1], reverse=True):
            pdf.add_paragraph(f"• {comp.replace('_', ' ').title()}: {score:.1f}/10")
        
        # Bias Analysis
        if assessment_results.get("bias_detected"):
            pdf.add_section("Bias Analysis")
            pdf.add_paragraph("The following biases were detected and corrected:")
            for bias in assessment_results["bias_detected"]:
                pdf.add_paragraph(f"• {bias}")
        
        # Recommendations
        pdf.add_section("Development Recommendations")
        for i, rec in enumerate(assessment_results.get("recommendations", []), 1):
            pdf.add_paragraph(f"{i}. {rec}")
        
        # Save PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"leadership_report_{timestamp}.pdf"
        filepath = self.output_path / filename
        pdf.output(str(filepath))
        
        return str(filepath)
    
    def _generate_executive_summary(self, results: Dict) -> str:
        """Generate executive summary text"""
        
        inclusive_score = results.get("inclusive_score", 0)
        traditional_score = results.get("traditional_score", 0)
        
        if inclusive_score >= 8:
            level = "exceptional"
        elif inclusive_score >= 6:
            level = "strong"
        else:
            level = "developing"
        
        summary = f"""
        This assessment reveals {level} leadership capabilities with particular strengths 
        in AI-era competencies. The inclusive assessment score ({inclusive_score:.1f}/10) 
        {'exceeds' if inclusive_score > traditional_score else 'aligns with'} the 
        traditional assessment ({traditional_score:.1f}/10), indicating 
        {'strong potential for modern, inclusive leadership' if inclusive_score > traditional_score else 'balanced leadership capabilities'}.
        """
        
        return summary.strip()
    
    def _calculate_ai_readiness(self, results: Dict) -> float:
        """Calculate AI-era leadership readiness score"""
        
        ai_competencies = ["empathy", "collaboration", "adaptability", "innovation", "continuous_learning"]
        competencies = results.get("competencies", {})
        
        scores = [competencies.get(comp, 5) for comp in ai_competencies]
        return (sum(scores) / len(scores)) * 10 if scores else 50
    
    def create_visualizations(self, results: Dict) -> Dict[str, str]:
        """Create all visualizations for the report"""
        
        visualizations = {}
        
        # Competency radar chart
        visualizations["competency_radar"] = self._create_competency_radar(results.get("competencies", {}))
        
        # Score comparison
        visualizations["score_comparison"] = self._create_score_comparison(
            results.get("traditional_score", 0),
            results.get("inclusive_score", 0)
        )
        
        # Leadership style distribution
        visualizations["leadership_styles"] = self._create_style_chart(results.get("leadership_scores", {}))
        
        return visualizations
    
    def _create_competency_radar(self, competencies: Dict[str, float]) -> str:
        """Create radar chart for competencies"""
        
        categories = list(competencies.keys())
        values = list(competencies.values())
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=[c.replace('_', ' ').title() for c in categories] + [categories[0].replace('_', ' ').title()],
            fill='toself',
            name='Current Profile'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=False,
            title="Leadership Competency Profile"
        )
        
        # Save to temporary file and return path
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.write_image(temp_file.name)
        return temp_file.name
    
    def _create_score_comparison(self, traditional: float, inclusive: float) -> str:
        """Create bar chart comparing scores"""
        
        fig = go.Figure(data=[
            go.Bar(x=['Traditional', 'Inclusive'], 
                   y=[traditional, inclusive],
                   marker_color=['lightblue', 'lightgreen'])
        ])
        
        fig.update_layout(
            title="Assessment Score Comparison",
            yaxis_title="Score (0-10)",
            showlegend=False
        )
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.write_image(temp_file.name)
        return temp_file.name
    
    def _create_style_chart(self, styles: Dict[str, float]) -> str:
        """Create horizontal bar chart for leadership styles"""
        
        style_names = [s.replace('_', ' ').title() for s in styles.keys()]
        values = list(styles.values())
        
        fig = go.Figure(data=[
            go.Bar(x=values, y=style_names, orientation='h',
                   marker_color='lightsalmon')
        ])
        
        fig.update_layout(
            title="Leadership Style Distribution",
            xaxis_title="Score (0-10)",
            showlegend=False
        )
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.write_image(temp_file.name)
        return temp_file.name


class CustomPDF(FPDF):
    """Custom PDF class with consistent formatting"""
    
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Empathic Leadership AI", 0, 1, "C")
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")
    
    def add_section(self, title: str):
        self.set_font("Arial", "B", 16)
        self.ln(10)
        self.cell(0, 10, title, ln=True)
        self.set_font("Arial", "", 12)
        self.ln(5)
    
    def add_paragraph(self, text: str):
        self.multi_cell(0, 10, text)
        self.ln(2)
