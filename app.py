#!/usr/bin/env python3
"""
Empathic Leadership AI - Integrated Application with ML Features
Complete integration of multimodal processing, ML model selection, and data collection
"""

import gradio as gr
import os
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import tempfile
import numpy as np
from pathlib import Path

# Import ML components
from src.ml_enhanced_integration import (
    EnhancedLeadershipAnalyzer,
    MultiModelManager,
    DataCollector
)

# Import existing components (we'll integrate them)
from src.multimodal_processor import MultimodalProcessor
from src.bias_detector import BiasDetector
from src.report_generator import ReportGenerator

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ Warning: OPENAI_API_KEY not set. Will use open-source models only.")

# Custom CSS for professional styling
css = """
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}
.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
}
.main-header h1 {
    margin: 0;
    font-size: 2.5em;
}
.main-header p {
    margin: 10px 0 0 0;
    opacity: 0.9;
}
.model-status {
    background: #f0f4f8;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.results-section {
    background: #ffffff;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin: 15px 0;
}
.bias-alert {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
"""

class IntegratedLeadershipAnalyzer:
    """
    Main analyzer that combines all functionality:
    - ML model selection and management
    - Multimodal processing
    - Bias detection and correction
    - Data collection for training
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize ML components
        self.ml_analyzer = EnhancedLeadershipAnalyzer(openai_api_key)
        
        # Initialize other processors
        self.multimodal_processor = MultimodalProcessor(openai_api_key)
        self.bias_detector = BiasDetector()
        self.report_generator = ReportGenerator()
        
        # Track session data
        self.current_assessment_id = None
        
    def analyze_leadership(
        self,
        text_input: str,
        audio_file: Optional[str] = None,
        video_file: Optional[str] = None,
        questionnaire_json: str = "{}",
        company_criteria_json: str = "{}",
        task_complexity: str = "Medium",
        save_for_training: bool = True,
        use_ensemble: bool = False
    ) -> Dict[str, Any]:
        """
        Complete leadership analysis with ML model selection
        """
        
        # Parse inputs
        questionnaire = json.loads(questionnaire_json) if questionnaire_json else {}
        company_criteria = json.loads(company_criteria_json) if company_criteria_json else {}
        
        # Process multimodal inputs
        combined_text = text_input
        
        if audio_file:
            audio_text = self.multimodal_processor.process_audio(audio_file)
            combined_text += f"\n\n[Audio Transcript]: {audio_text}"
            
        if video_file:
            video_analysis = self.multimodal_processor.process_video(video_file)
            combined_text += f"\n\n[Video Analysis]: {video_analysis}"
            
        if questionnaire:
            q_text = self.multimodal_processor.process_questionnaire(questionnaire)
            combined_text += f"\n\n[Questionnaire Responses]: {q_text}"
        
        # Run ML-enhanced analysis
        ml_result = self.ml_analyzer.analyze_with_ml(
            text=combined_text,
            task_complexity=task_complexity.lower(),
            save_for_training=save_for_training
        )
        
        # Extract and enhance results
        result = {
            "model_used": ml_result["model_used"],
            "model_type": ml_result["model_type"],
            "ml_confidence": ml_result.get("ml_confidence", 0.85),
            "leadership_scores": ml_result["leadership_scores"],
            "competencies": ml_result["competencies"],
            "bias_detected": ml_result.get("bias_detected", []),
            "ml_stats": ml_result["ml_stats"],
            "timestamp": datetime.now()
        }
        
        # Add bias detection
        bias_analysis = self.bias_detector.detect_biases(combined_text)
        result["bias_detected"].extend(bias_analysis.get("biases", []))
        result["bias_corrections"] = bias_analysis.get("corrections", {})
        
        # Calculate traditional vs inclusive scores
        result["traditional_score"] = self._calculate_traditional_score(result["competencies"])
        result["inclusive_score"] = self._calculate_inclusive_score(
            result["competencies"], 
            result["bias_corrections"]
        )
        
        # Generate recommendations
        result["recommendations"] = self._generate_recommendations(result)
        
        # Store assessment ID for feedback
        self.current_assessment_id = ml_result.get("assessment_id", "unknown")
        
        return result
    
    def _calculate_traditional_score(self, competencies: Dict[str, float]) -> float:
        """Calculate traditional leadership score"""
        traditional_weights = {
            "strategic_thinking": 0.2,
            "decision_making": 0.2,
            "communication": 0.15,
            "innovation": 0.15,
            "team_building": 0.1,
            "conflict_resolution": 0.1,
            "adaptability": 0.05,
            "empathy": 0.03,
            "collaboration": 0.02
        }
        
        score = sum(
            competencies.get(comp, 5) * weight 
            for comp, weight in traditional_weights.items()
        )
        return min(score, 10)
    
    def _calculate_inclusive_score(self, competencies: Dict[str, float], corrections: Dict) -> float:
        """Calculate inclusive leadership score with bias corrections"""
        inclusive_weights = {
            "empathy": 0.2,
            "collaboration": 0.2,
            "adaptability": 0.15,
            "communication": 0.15,
            "innovation": 0.1,
            "strategic_thinking": 0.1,
            "team_building": 0.05,
            "decision_making": 0.03,
            "conflict_resolution": 0.02
        }
        
        # Apply corrections
        corrected_competencies = competencies.copy()
        for comp, correction in corrections.items():
            if comp in corrected_competencies:
                corrected_competencies[comp] *= (1 + correction)
        
        score = sum(
            corrected_competencies.get(comp, 5) * weight 
            for comp, weight in inclusive_weights.items()
        )
        return min(score, 10)
    
    def _generate_recommendations(self, result: Dict) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        # Analyze weak areas
        weak_competencies = [
            comp for comp, score in result["competencies"].items() 
            if score < 6
        ]
        
        # Add specific recommendations
        if "empathy" in weak_competencies:
            recommendations.append(
                "🎯 Develop emotional intelligence through active listening exercises and empathy mapping workshops"
            )
        
        if "strategic_thinking" in weak_competencies:
            recommendations.append(
                "📊 Enhance strategic thinking with scenario planning and systems thinking courses"
            )
            
        if result["bias_detected"]:
            recommendations.append(
                "🌈 Participate in unconscious bias training and inclusive leadership workshops"
            )
            
        # Model-specific recommendations
        if result["model_type"] == "openai":
            recommendations.append(
                f"💡 Based on {result['model_used']} analysis: Focus on AI-era competencies"
            )
            
        return recommendations if recommendations else [
            "✨ Continue developing your balanced leadership approach",
            "📚 Consider advanced leadership coaching for next-level growth"
        ]

# Initialize global analyzer
analyzer = IntegratedLeadershipAnalyzer(OPENAI_API_KEY)

# UI Helper Functions
def create_competency_radar_chart(competencies: Dict[str, float]) -> go.Figure:
    """Create an enhanced radar chart for competencies"""
    
    categories = [k.replace('_', ' ').title() for k in competencies.keys()]
    values = list(competencies.values())
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    # Add main trace
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Profile',
        fillcolor='rgba(103, 126, 234, 0.3)',
        line=dict(color='#667eea', width=2)
    ))
    
    # Add benchmark trace (AI-era ideal)
    benchmark_values = [8, 8, 9, 9, 8, 8, 9, 9, 9, 8]  # High in empathy, collaboration
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values,
        theta=categories,
        name='AI-Era Benchmark',
        line=dict(color='#764ba2', dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickmode='linear',
                tick0=0,
                dtick=2
            )
        ),
        showlegend=True,
        title={
            'text': "Leadership Competencies Profile",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=450
    )
    
    return fig

def create_model_usage_chart(ml_stats: Dict) -> go.Figure:
    """Create model usage visualization"""
    
    usage_data = ml_stats["model_usage"]
    
    models = []
    used = []
    remaining = []
    colors = []
    
    for model, stats in usage_data.items():
        if stats["daily_limit"] != float('inf'):
            models.append(model)
            used_pct = stats["percentage_used"]
            used.append(used_pct)
            remaining.append(100 - used_pct)
            
            # Color coding
            if used_pct > 80:
                colors.append('#FF6B6B')
            elif used_pct > 50:
                colors.append('#FFD93D')
            else:
                colors.append('#6BCF7F')
    
    fig = go.Figure()
    
    # Stacked bar chart
    fig.add_trace(go.Bar(
        name='Used',
        x=models,
        y=used,
        marker_color=colors,
        text=[f'{u:.0f}%' for u in used],
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        name='Remaining',
        x=models,
        y=remaining,
        marker_color='lightgray',
        text=[f'{r:.0f}%' for r in remaining],
        textposition='inside'
    ))
    
    fig.update_layout(
        barmode='stack',
        title="Model Usage Status (Daily Limits)",
        yaxis_title="Percentage",
        height=300,
        showlegend=False
    )
    
    return fig

def create_bias_comparison_chart(traditional: float, inclusive: float, biases: List[str]) -> go.Figure:
    """Create comparison chart with bias indicators"""
    
    fig = go.Figure()
    
    # Bars
    fig.add_trace(go.Bar(
        x=['Traditional Assessment', 'Inclusive Assessment'],
        y=[traditional, inclusive],
        marker_color=['#FF6B6B', '#6BCF7F'],
        text=[f'{traditional:.1f}/10', f'{inclusive:.1f}/10'],
        textposition='auto',
        hovertemplate='%{x}: %{y:.1f}/10<extra></extra>'
    ))
    
    # Add bias annotations if detected
    if biases:
        fig.add_annotation(
            x=0, y=traditional + 0.5,
            text=f"⚠️ {len(biases)} bias(es) detected",
            showarrow=False,
            font=dict(color="red", size=12)
        )
    
    fig.update_layout(
        title="Assessment Score Comparison",
        yaxis_title="Score",
        yaxis=dict(range=[0, 11]),
        showlegend=False,
        height=350
    )
    
    return fig

# Main processing function
def process_leadership_assessment(
    text_input: str,
    audio_file: Optional[str],
    video_file: Optional[str],
    questionnaire_json: str,
    company_criteria_json: str,
    complexity: str,
    save_data: bool,
    use_ensemble: bool
) -> Tuple[Any, ...]:
    """
    Main processing function that returns all UI components
    """
    
    try:
        # Run analysis
        result = analyzer.analyze_leadership(
            text_input=text_input,
            audio_file=audio_file,
            video_file=video_file,
            questionnaire_json=questionnaire_json,
            company_criteria_json=company_criteria_json,
            task_complexity=complexity,
            save_for_training=save_data,
            use_ensemble=use_ensemble
        )
        
        # Create visualizations
        competency_chart = create_competency_radar_chart(result["competencies"])
        model_usage_chart = create_model_usage_chart(result["ml_stats"])
        comparison_chart = create_bias_comparison_chart(
            result["traditional_score"],
            result["inclusive_score"],
            result["bias_detected"]
        )
        
        # Format main analysis
        analysis_text = f"""
        <div class="model-status">
        <h3>🤖 Model Information</h3>
        <p><strong>Model Used:</strong> {result['model_used']}</p>
        <p><strong>Model Type:</strong> {result['model_type']}</p>
        <p><strong>Confidence:</strong> {result['ml_confidence']:.0%}</p>
        </div>
        
        <div class="results-section">
        <h3>📊 Assessment Results</h3>
        <p><strong>Traditional Score:</strong> {result['traditional_score']:.1f}/10</p>
        <p><strong>Inclusive Score:</strong> {result['inclusive_score']:.1f}/10</p>
        
        <h4>Leadership Style Analysis</h4>
        """
        
        for style, score in result["leadership_scores"].items():
            analysis_text += f"<p>• {style.replace('_', ' ').title()}: {score:.1f}/10</p>"
        
        analysis_text += "</div>"
        
        # Format recommendations
        recommendations_html = "<div class='results-section'><h3>🚀 Development Recommendations</h3><ol>"
        for rec in result["recommendations"]:
            recommendations_html += f"<li>{rec}</li>"
        recommendations_html += "</ol></div>"
        
        # Format bias detection
        if result["bias_detected"]:
            bias_html = f"""
            <div class='bias-alert'>
            <h3>⚠️ Bias Detection Alert</h3>
            <p>The following potential biases were detected in the assessment:</p>
            <ul>
            """
            for bias in result["bias_detected"]:
                bias_html += f"<li>{bias}</li>"
            bias_html += """
            </ul>
            <p><em>The inclusive score has been adjusted to compensate for these biases.</em></p>
            </div>
            """
        else:
            bias_html = """
            <div class='results-section'>
            <h3>✅ Bias Detection</h3>
            <p>No significant biases detected in this assessment.</p>
            </div>
            """
        
        # ML status
        ml_stats = result["ml_stats"]
        ml_status_html = f"""
        <div class='model-status'>
        <h3>📊 ML System Status</h3>
        <p><strong>Assessments Collected:</strong> {ml_stats['data_collected']['total_assessments']}</p>
        <p><strong>Average Rating:</strong> {ml_stats['data_collected']['average_rating']}/5</p>
        <p><strong>Training Ready:</strong> {ml_stats['can_train'][1]}</p>
        <p><strong>Assessment ID:</strong> <code>{analyzer.current_assessment_id}</code></p>
        </div>
        """
        
        return (
            competency_chart,
            comparison_chart,
            model_usage_chart,
            gr.HTML(analysis_text),
            gr.HTML(recommendations_html),
            gr.HTML(bias_html),
            gr.HTML(ml_status_html),
            analyzer.current_assessment_id  # For feedback
        )
        
    except Exception as e:
        error_html = f"""
        <div style='background: #fee; padding: 20px; border-radius: 10px; color: #c00;'>
        <h3>❌ Error Processing Assessment</h3>
        <p>{str(e)}</p>
        <p>Please check your inputs and try again.</p>
        </div>
        """
        empty_fig = go.Figure()
        return (
            empty_fig, empty_fig, empty_fig,
            gr.HTML(error_html), gr.HTML(""), gr.HTML(""), gr.HTML(""),
            ""
        )

# Feedback function
def submit_feedback(assessment_id: str, rating: int, feedback_text: str) -> str:
    """Submit user feedback for an assessment"""
    
    if not assessment_id or assessment_id == "unknown":
        return "❌ Invalid assessment ID. Please run an analysis first."
    
    try:
        analyzer.ml_analyzer.add_user_feedback(assessment_id, rating, feedback_text)
        return f"""
        <div style='background: #dfd; padding: 15px; border-radius: 8px;'>
        ✅ <strong>Feedback recorded successfully!</strong><br>
        Your rating of {rating}/5 and feedback will help improve our models.<br>
        Thank you for contributing to better leadership assessment!
        </div>
        """
    except Exception as e:
        return f"❌ Error recording feedback: {str(e)}"

# Training trigger function
def trigger_model_training() -> str:
    """Trigger ML model training"""
    
    try:
        result = analyzer.ml_analyzer.trigger_training()
        
        if result["success"]:
            return f"""
            <div style='background: #dfd; padding: 20px; border-radius: 10px;'>
            <h3>✅ Training Completed Successfully!</h3>
            
            <h4>📊 Training Metrics:</h4>
            <ul>
            <li>Accuracy: {result['training_metrics']['metrics']['accuracy']:.2%}</li>
            <li>Bias Detection F1: {result['training_metrics']['metrics']['bias_detection_f1']:.2%}</li>
            <li>Loss: {result['training_metrics']['metrics']['loss']:.3f}</li>
            </ul>
            
            <h4>🎯 Improvements:</h4>
            <ul>
            """
            
            for improvement in result['training_metrics']['improvements']:
                result += f"<li>{improvement}</li>"
                
            result += """
            </ul>
            <p><em>The new model will be used for future assessments.</em></p>
            </div>
            """
            return result
        else:
            return f"""
            <div style='background: #ffd; padding: 20px; border-radius: 10px;'>
            <h3>⚠️ Cannot Train Yet</h3>
            <p>{result['message']}</p>
            <p>Keep collecting more assessments to enable training!</p>
            </div>
            """
    except Exception as e:
        return f"❌ Error during training: {str(e)}"

# Sample data generators
def create_sample_questionnaire() -> str:
    """Generate sample questionnaire"""
    return json.dumps({
        "leadership_philosophy": "I believe in servant leadership, putting my team's growth first",
        "decision_style": "Collaborative with data-driven final decisions",
        "conflict_approach": "Direct communication with empathy and active listening",
        "innovation_mindset": "Encourage experimentation and learn from failures",
        "diversity_commitment": "Actively build inclusive teams with diverse perspectives"
    }, indent=2)

def create_sample_criteria() -> str:
    """Generate sample company criteria"""
    return json.dumps({
        "strategic_thinking": 8,
        "innovation": 9,
        "empathy": 8,
        "collaboration": 9,
        "adaptability": 8
    }, indent=2)

# Build the Gradio interface
with gr.Blocks(css=css, title="Empathic Leadership AI", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🌍 Empathic Leadership AI</h1>
        <p>ML-Enhanced Framework for Inclusive Leadership Assessment</p>
    </div>
    """)
    
    # Main interface
    with gr.Tabs():
        # Analysis Tab
        with gr.Tab("🔍 Leadership Analysis"):
            with gr.Row():
                # Input column
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Input Your Leadership Data")
                    
                    text_input = gr.Textbox(
                        label="Leadership Description",
                        placeholder="Describe your leadership style, experiences, and philosophy...",
                        lines=6,
                        value="I believe in creating psychologically safe environments where team members can experiment, fail, and learn. My leadership approach combines data-driven decision making with deep empathy for individual needs and circumstances."
                    )
                    
                    with gr.Row():
                        audio_input = gr.Audio(
                            label="Audio Recording (Optional)",
                            type="filepath"
                        )
                        
                        video_input = gr.File(
                            label="Video File (Optional)",
                            file_types=[".mp4", ".avi", ".mov"]
                        )
                    
                    with gr.Accordion("📋 Structured Questionnaire", open=False):
                        questionnaire_input = gr.Code(
                            label="Questionnaire Responses (JSON)",
                            language="json",
                            value=create_sample_questionnaire()
                        )
                    
                    with gr.Accordion("🏢 Company Criteria", open=False):
                        criteria_input = gr.Code(
                            label="Role Requirements (JSON)",
                            language="json",
                            value=create_sample_criteria()
                        )
                    
                    gr.Markdown("### ⚙️ Analysis Settings")
                    
                    complexity = gr.Radio(
                        ["Simple", "Medium", "Complex"],
                        label="Analysis Depth",
                        value="Medium",
                        info="Complex analysis uses more advanced models"
                    )
                    
                    with gr.Row():
                        save_data = gr.Checkbox(
                            label="Save for ML Training",
                            value=True,
                            info="Help improve our models"
                        )
                        
                        use_ensemble = gr.Checkbox(
                            label="Use Model Ensemble",
                            value=False,
                            info="Combine multiple models (slower)"
                        )
                    
                    analyze_btn = gr.Button(
                        "🚀 Analyze Leadership Profile",
                        variant="primary",
                        size="lg"
                    )
                
                # Results column
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Analysis Results")
                    
                    # Charts row
                    with gr.Row():
                        competency_chart = gr.Plot(label="Competency Profile")
                        comparison_chart = gr.Plot(label="Score Comparison")
                    
                    model_usage_chart = gr.Plot(label="Model Usage Status")
                    
                    # Analysis sections
                    analysis_output = gr.HTML(label="Analysis Results")
                    recommendations_output = gr.HTML(label="Recommendations")
                    bias_output = gr.HTML(label="Bias Detection")
                    ml_status_output = gr.HTML(label="ML Status")
                    
                    # Hidden assessment ID
                    assessment_id_store = gr.Textbox(visible=False)
            
            # Connect analyze button
            analyze_btn.click(
                fn=process_leadership_assessment,
                inputs=[
                    text_input,
                    audio_input,
                    video_input,
                    questionnaire_input,
                    criteria_input,
                    complexity,
                    save_data,
                    use_ensemble
                ],
                outputs=[
                    competency_chart,
                    comparison_chart,
                    model_usage_chart,
                    analysis_output,
                    recommendations_output,
                    bias_output,
                    ml_status_output,
                    assessment_id_store
                ]
            )
        
        # Feedback Tab
        with gr.Tab("💬 Provide Feedback"):
            gr.Markdown("""
            ### Help Improve Our Models
            
            Your feedback helps us create more accurate and inclusive assessments.
            Rate your analysis and provide specific feedback about what could be improved.
            """)
            
            with gr.Row():
                with gr.Column():
                    feedback_id = gr.Textbox(
                        label="Assessment ID",
                        placeholder="Automatically filled after analysis",
                        interactive=True
                    )
                    
                    rating = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=4,
                        label="Rating (1-5 stars)",
                        info="How accurate was the assessment?"
                    )
                    
                    feedback_text = gr.Textbox(
                        label="Detailed Feedback",
                        placeholder="What was accurate? What could be improved? Any biases we missed?",
                        lines=4
                    )
                    
                    submit_feedback_btn = gr.Button("Submit Feedback", variant="primary")
                
                with gr.Column():
                    feedback_result = gr.HTML()
            
            # Auto-fill assessment ID
            assessment_id_store.change(
                fn=lambda x: x,
                inputs=assessment_id_store,
                outputs=feedback_id
            )
            
            submit_feedback_btn.click(
                fn=submit_feedback,
                inputs=[feedback_id, rating, feedback_text],
                outputs=feedback_result
            )
        
        # ML Training Tab
        with gr.Tab("🎓 ML Training"):
            gr.Markdown("""
            ### Train Custom Models
            
            Use collected assessment data to improve bias detection and leadership analysis.
            Training requires at least 10 high-quality assessments (rated 4+ stars).
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    #### Training Process:
                    1. Collects high-rated assessments
                    2. Trains bias detection patterns
                    3. Improves competency predictions
                    4. Validates on test data
                    5. Deploys improved model
                    
                    **Note:** This is a simulated training for the MVP.
                    Real training would use actual ML pipelines.
                    """)
                    
                    train_btn = gr.Button(
                        "🚀 Start Training",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column():
                    training_result = gr.HTML()
            
            train_btn.click(
                fn=trigger_model_training,
                outputs=training_result
            )
        
        # About Tab
        with gr.Tab("📖 About"):
            gr.Markdown("""
            ## About Empathic Leadership AI
            
            This ML-enhanced framework provides inclusive leadership assessment by:
            
            ### 🤖 Smart Model Selection
            - Automatically chooses the best available model based on task complexity
            - Manages API limits across multiple models (GPT-4.5, GPT-4o, o1, open-source)
            - Falls back to open-source models when limits are reached
            
            ### 🌈 Inclusive Assessment
            - Detects and corrects gender and neurodiversity biases
            - Emphasizes AI-era competencies (empathy, collaboration)
            - Provides both traditional and inclusive scoring
            
            ### 📊 Continuous Learning
            - Collects anonymized assessment data
            - Learns from user feedback
            - Improves bias detection over time
            - Trains custom models on your organization's data
            
            ### 🔬 Research Foundation
            Built on extensive research in:
            - Leadership theory (Bass, Goleman, Hersey-Blanchard)
            - Gender studies and double-bind theory
            - Neurodiversity and inclusive practices
            - AI ethics and algorithmic fairness
            
            ### 🚀 Future Roadmap
            - Real-time video analysis
            - Multi-language support
            - Team assessment capabilities
            - API for integration
            - Longitudinal tracking
            
            **Open Source:** MIT Licensed | [GitHub](https://github.com/your-username/empathic-leadership-ai)
            """)

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
