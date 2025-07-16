#!/usr/bin/env python3
"""
Empathic Leadership AI - Simplified Gradio App for HF Spaces
"""

import gradio as gr
import os
import json
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from openai import OpenAI

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in Space secrets")

client = OpenAI(api_key=OPENAI_API_KEY)

# Leadership competencies
COMPETENCIES = [
    "Strategic Thinking",
    "Decision Making", 
    "Communication",
    "Team Building",
    "Conflict Resolution",
    "Adaptability",
    "Innovation",
    "Empathy",
    "Collaboration"
]

def analyze_leadership_text(text: str, additional_context: Dict = None) -> Dict:
    """Analyze leadership text using OpenAI API"""
    
    prompt = f"""
    Analyze the following leadership description and rate the person on these competencies (1-10 scale):
    {', '.join(COMPETENCIES)}
    
    Also identify:
    1. Leadership style (transformational, situational, servant, etc.)
    2. Emotional intelligence components
    3. Potential gender or neurodiversity biases in traditional assessment
    4. Recommendations for development
    
    Leadership description: {text}
    
    Return as JSON with structure:
    {{
        "competencies": {{"Strategic Thinking": 8, ...}},
        "leadership_styles": {{"transformational": 7, ...}},
        "emotional_intelligence": {{"self_awareness": 8, ...}},
        "traditional_score": 7.5,
        "inclusive_score": 8.2,
        "bias_detected": ["list of biases"],
        "recommendations": ["list of recommendations"]
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in inclusive leadership assessment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "error": str(e),
            "competencies": {comp: 5 for comp in COMPETENCIES},
            "traditional_score": 5,
            "inclusive_score": 5,
            "recommendations": ["Error in analysis"]
        }

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
    """Create a comparison chart"""
    
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
    questionnaire_json: str = "{}"
) -> Tuple[go.Figure, go.Figure, str, str, str]:
    """Main processing function"""
    
    try:
        # Parse questionnaire
        questionnaire = json.loads(questionnaire_json) if questionnaire_json else {}
        
        # Combine text and questionnaire
        full_text = text_input
        if questionnaire:
            full_text += "\n\nQuestionnaire responses:\n"
            for q, a in questionnaire.items():
                full_text += f"{q}: {a}\n"
        
        # Analyze
        result = analyze_leadership_text(full_text)
        
        # Create visualizations
        competency_chart = create_competency_radar_chart(result.get("competencies", {}))
        comparison_chart = create_comparison_chart(
            result.get("traditional_score", 5),
            result.get("inclusive_score", 5)
        )
        
        # Format results
        analysis_text = f"""
        ## 📊 Assessment Results
        
        **Traditional Assessment Score:** {result.get('traditional_score', 5):.1f}/10
        **Inclusive Assessment Score:** {result.get('inclusive_score', 5):.1f}/10
        
        ### 🎯 Leadership Styles
        """
        
        for style, score in result.get("leadership_styles", {}).items():
            analysis_text += f"- **{style.replace('_', ' ').title()}:** {score}/10\n"
        
        # Recommendations
        recommendations_text = "## 🚀 Development Recommendations\n\n"
        for i, rec in enumerate(result.get("recommendations", []), 1):
            recommendations_text += f"{i}. {rec}\n"
        
        # Bias detection
        bias_text = "## 🔍 Bias Detection Results\n\n"
        biases = result.get("bias_detected", [])
        if biases:
            bias_text += "**Detected Potential Biases:**\n"
            for bias in biases:
                bias_text += f"⚠️ {bias}\n"
        else:
            bias_text += "✅ No significant biases detected."
        
        return competency_chart, comparison_chart, analysis_text, recommendations_text, bias_text
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        empty_fig = go.Figure()
        return empty_fig, empty_fig, error_msg, error_msg, error_msg

# Gradio interface
with gr.Blocks(title="Empathic Leadership AI") as demo:
    gr.Markdown("""
    # 🌍 Empathic Leadership AI
    ### Open-source framework for inclusive leadership assessment
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Leadership Description",
                placeholder="Describe your leadership style...",
                lines=10,
                value="I believe in creating an inclusive environment where every team member feels valued and empowered to contribute their unique perspectives."
            )
            
            questionnaire_input = gr.Textbox(
                label="Questionnaire (JSON)",
                placeholder='{"question": "answer"}',
                lines=3,
                value='{}'
            )
            
            analyze_btn = gr.Button("🔍 Analyze Leadership", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Row():
                competency_chart = gr.Plot(label="Competency Profile")
                comparison_chart = gr.Plot(label="Assessment Comparison")
            
            analysis_output = gr.Markdown(label="Analysis")
            recommendations_output = gr.Markdown(label="Recommendations")
            bias_output = gr.Markdown(label="Bias Detection")
    
    analyze_btn.click(
        fn=process_leadership_assessment,
        inputs=[text_input, questionnaire_input],
        outputs=[
            competency_chart,
            comparison_chart,
            analysis_output,
            recommendations_output,
            bias_output
        ]
    )
    
    gr.Markdown("""
    ---
    ### About
    This tool provides inclusive, bias-aware leadership assessment using AI.
    It corrects for gender and neurodiversity biases in traditional evaluations.
    
    **Note:** Set your OpenAI API key in Space secrets for full functionality.
    """)

if __name__ == "__main__":
    demo.launch()
