#!/usr/bin/env python3
"""
Empathic Leadership AI - Main Gradio Application
Hugging Face Spaces deployment
"""

import gradio as gr
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import tempfile
import io
import base64

# Import our modules
from src.leadership_analyzer import EmpathicLeadershipAI
from src.report_generator import ReportGenerator

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize the analyzer
analyzer = EmpathicLeadershipAI(OPENAI_API_KEY)
report_gen = ReportGenerator()

# Custom CSS for better styling
css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    color: #2E5CE6;
    margin-bottom: 20px;
}
.competency-chart {
    height: 400px;
}
.results-section {
    background: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
"""

def create_competency_radar_chart(competencies: Dict[str, float]) -> go.Figure:
    """Create a radar chart for competencies"""
    
    categories = list(competencies.keys())
    values = list(competencies.values())
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Competencies',
        line=dict(color='#2E5CE6'),
        fillcolor='rgba(46, 92, 230, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        showlegend=True,
        title="Leadership Competencies Profile",
        height=400
    )
    
    return fig

def create_comparison_chart(traditional_score: float, inclusive_score: float) -> go.Figure:
    """Create a comparison chart between traditional and inclusive scores"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Traditional Assessment', 'Inclusive Assessment'],
        y=[traditional_score, inclusive_score],
        marker_color=['#FF6B6B', '#4ECDC4'],
        text=[f'{traditional_score:.1f}/10', f'{inclusive_score:.1f}/10'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Assessment Comparison",
        yaxis_title="Score",
        yaxis=dict(range=[0, 10]),
        showlegend=False,
        height=300
    )
    
    return fig

def process_leadership_assessment(
    text_input: str,
    audio_file: Optional[str] = None,
    video_file: Optional[str] = None,
    questionnaire_json: str = "{}",
    company_criteria_json: str = "{}",
    assessment_focus: str = "Balanced"
) -> Tuple[go.Figure, go.Figure, str, str, str]:
    """
    Main processing function for leadership assessment
    
    Returns:
        - Competency radar chart
        - Comparison chart
        - Analysis text
        - Recommendations
        - Bias detection results
    """
    
    try:
        # Parse JSON inputs
        questionnaire = json.loads(questionnaire_json) if questionnaire_json else {}
        company_criteria = json.loads(company_criteria_json) if company_criteria_json else {}
        
        # Process multimodal input
        result = analyzer.process_multimodal_input(
            audio_path=audio_file,
            video_path=video_file,
            text=text_input,
            questionnaire=questionnaire
        )
        
        # Apply company criteria if provided
        if company_criteria:
            result = analyzer.analyze_text(text_input, company_criteria)
        
        # Create visualizations
        competency_chart = create_competency_radar_chart(result.competencies)
        comparison_chart = create_comparison_chart(result.traditional_score, result.inclusive_score)
        
        # Format analysis results
        analysis_text = f"""
        ## 📊 Assessment Results
        
        **Traditional Assessment Score:** {result.traditional_score:.1f}/10
        **Inclusive Assessment Score:** {result.inclusive_score:.1f}/10
        
        ### 🎯 Leadership Styles Distribution
        """
        
        for style, score in result.leadership_styles.items():
            analysis_text += f"- **{style.replace('_', ' ').title()}:** {score:.1f}/10\n"
        
        analysis_text += f"""
        
        ### 🧠 Emotional Intelligence Profile
        """
        
        for eq_component, score in result.emotional_intelligence.items():
            analysis_text += f"- **{eq_component.replace('_', ' ').title()}:** {score:.1f}/10\n"
        
        # Format recommendations
        recommendations_text = "## 🚀 Development Recommendations\n\n"
        for i, rec in enumerate(result.recommendations, 1):
            recommendations_text += f"{i}. {rec}\n"
        
        # Add cost estimation
        if result.cost_estimation.get('total', 0) > 0:
            recommendations_text += f"\n**💰 Estimated Investment:** ${result.cost_estimation['total']:,.0f}\n"
        
        # Format bias detection
        bias_text = "## 🔍 Bias Detection Results\n\n"
        if result.bias_detected:
            bias_text += "**Detected Potential Biases:**\n"
            for bias in result.bias_detected:
                bias_text += f"⚠️ {bias}\n"
        else:
            bias_text += "✅ No significant biases detected in the assessment."
        
        return competency_chart, comparison_chart, analysis_text, recommendations_text, bias_text
        
    except Exception as e:
        error_msg = f"Error processing assessment: {str(e)}"
        empty_fig = go.Figure()
        return empty_fig, empty_fig, error_msg, error_msg, error_msg

def create_sample_questionnaire() -> str:
    """Create a sample questionnaire JSON"""
    sample = {
        "leadership_philosophy": "I believe in empowering my team to make autonomous decisions while providing clear strategic direction.",
        "conflict_resolution": "I prefer to address conflicts directly through open dialogue and finding win-win solutions.",
        "team_motivation": "I motivate my team through recognition, growth opportunities, and connecting their work to larger purpose.",
        "decision_making": "I gather input from stakeholders but make final decisions based on data and strategic alignment.",
        "change_management": "I help teams navigate change by communicating the vision clearly and supporting them through transitions."
    }
    return json.dumps(sample, indent=2)

def create_sample_company_criteria() -> str:
    """Create sample company criteria JSON"""
    sample = {
        "strategic_thinking": 8,
        "innovation": 9,
        "empathy": 7,
        "decision_making": 8,
        "adaptability": 9
    }
    return json.dumps(sample, indent=2)

# Create the Gradio interface
with gr.Blocks(css=css, title="Empathic Leadership AI") as demo:
    gr.HTML("""
    <div class="main-header">
        <h1>🌍 Empathic Leadership AI</h1>
        <p>Open-source framework for inclusive leadership assessment</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📝 Input Data")
            
            # Text input
            text_input = gr.Textbox(
                label="Leadership Description",
                placeholder="Describe your leadership style, philosophy, or experiences...",
                lines=5,
                value="I believe in creating an inclusive environment where every team member feels valued and empowered to contribute their unique perspectives. I focus on building trust through transparency and consistent communication, while fostering innovation through collaborative problem-solving."
            )
            
            # Audio input
            audio_input = gr.Audio(
                label="Audio Interview/Presentation",
                type="filepath",
                optional=True
            )
            
            # Video input
            video_input = gr.File(
                label="Video Presentation",
                file_types=[".mp4", ".avi", ".mov"],
                optional=True
            )
            
            # Questionnaire input
            with gr.Accordion("📋 Questionnaire (JSON format)", open=False):
                questionnaire_input = gr.Textbox(
                    label="Questionnaire Responses",
                    placeholder="Enter questionnaire responses in JSON format",
                    lines=3,
                    value=create_sample_questionnaire()
                )
            
            # Company criteria
            with gr.Accordion("🏢 Company Criteria (Optional)", open=False):
                company_criteria_input = gr.Textbox(
                    label="Company-Specific Requirements",
                    placeholder="Enter company criteria in JSON format",
                    lines=3,
                    value=create_sample_company_criteria()
                )
            
            # Assessment focus
            assessment_focus = gr.Dropdown(
                label="Assessment Focus",
                choices=["Balanced", "Innovation-Focused", "Team-Oriented", "Strategic"],
                value="Balanced"
            )
            
            # Process button
            process_btn = gr.Button("🔍 Analyze Leadership Profile", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("## 📊 Results")
            
            # Charts
            with gr.Row():
                competency_chart = gr.Plot(label="Competency Profile")
                comparison_chart = gr.Plot(label="Assessment Comparison")
            
            # Analysis results
            with gr.Accordion("📈 Detailed Analysis", open=True):
                analysis_output = gr.Markdown(label="Analysis Results")
            
            # Recommendations
            with gr.Accordion("🚀 Development Recommendations", open=True):
                recommendations_output = gr.Markdown(label="Recommendations")
            
            # Bias detection
            with gr.Accordion("🔍 Bias Detection", open=True):
                bias_output = gr.Markdown(label="Bias Analysis")
    
    # Connect the process button
    process_btn.click(
        fn=process_leadership_assessment,
        inputs=[
            text_input,
            audio_input,
            video_input,
            questionnaire_input,
            company_criteria_input,
            assessment_focus
        ],
        outputs=[
            competency_chart,
            comparison_chart,
            analysis_output,
            recommendations_output,
            bias_output
        ]
    )
    
    # Add information tabs
    with gr.Tabs():
        with gr.Tab("📖 About"):
            gr.Markdown("""
            ## About Empathic Leadership AI
            
            This open-source framework addresses the evolving nature of leadership in the AI era by providing:
            
            - **Inclusive Assessment**: Recognizes diverse leadership styles beyond traditional hierarchical models
            - **Bias Correction**: Identifies and corrects gender and neurodiversity biases in evaluation
            - **AI-Era Competencies**: Emphasizes empathy, collaboration, and adaptability
            - **Actionable Insights**: Provides personalized development recommendations
            
            ### How it works:
            1. **Input Processing**: Analyzes text, audio, video, and questionnaire data
            2. **Multi-Framework Assessment**: Applies various leadership theories and EQ models
            3. **Bias Detection**: Identifies potential biases in traditional assessments
            4. **Inclusive Correction**: Adjusts scores to account for systemic biases
            5. **Recommendations**: Generates personalized development plans
            
            **Research Foundation**: Built on extensive research in leadership theory, emotional intelligence, gender studies, and neurodiversity.
            """)
        
        with gr.Tab("🔬 Methodology"):
            gr.Markdown("""
            ## Assessment Methodology
            
            ### Leadership Frameworks
            - **Transformational Leadership** (Bass & Riggio)
            - **Situational Leadership** (Hersey & Blanchard)
            - **Emotional Intelligence** (Goleman)
            - **Authentic Leadership** (Avolio & Gardner)
            
            ### Core Competencies
            - Strategic Thinking
            - Decision Making
            - Communication
            - Team Building
            - Conflict Resolution
            - Adaptability
            - Innovation
            - Empathy
            - Collaboration
            
            ### Bias Correction
            - **Double Bind Detection**: Identifies "too soft" vs "too aggressive" patterns
            - **Gender Stereotype Correction**: Adjusts for systemic evaluation biases
            - **Neurodiversity Awareness**: Recognizes alternative leadership expressions
            - **Cultural Sensitivity**: Accounts for diverse leadership traditions
            """)
        
        with gr.Tab("💡 Usage Tips"):
            gr.Markdown("""
            ## How to Use This Tool
            
            ### Input Guidelines
            - **Text**: Provide detailed descriptions of your leadership philosophy and experiences
            - **Audio**: Upload clear recordings of presentations or interviews
            - **Questionnaire**: Use structured responses for comprehensive assessment
            - **Company Criteria**: Specify role-specific requirements for targeted evaluation
            
            ### Interpreting Results
            - **Traditional vs Inclusive Scores**: Compare how different methodologies evaluate your leadership
            - **Competency Radar**: Identify strengths and development areas
            - **Bias Detection**: Understand potential evaluation biases
            - **Recommendations**: Follow personalized development suggestions
            
            ### Best Practices
            - Be authentic and detailed in your responses
            - Consider the context of your leadership role
            - Use results as a starting point for development conversations
            - Regularly reassess to track progress
            """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
