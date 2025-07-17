# src/leadership_analyzer.py
"""
Core Leadership Analysis Module
Provides traditional analysis capabilities (for backward compatibility)
"""

import json
from typing import Dict, List, Optional, Any
import openai
from datetime import datetime

class EmpathicLeadershipAI:
    """
    Legacy analyzer class for backward compatibility
    New code should use EnhancedLeadershipAnalyzer from ml_enhanced_integration
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        if api_key:
            openai.api_key = api_key
        
        # Leadership frameworks
        self.frameworks = self._load_frameworks()
        
    def _load_frameworks(self) -> Dict[str, Any]:
        """Load leadership assessment frameworks"""
        
        return {
            "transformational": {
                "components": [
                    "Idealized Influence",
                    "Inspirational Motivation", 
                    "Intellectual Stimulation",
                    "Individualized Consideration"
                ]
            },
            "emotional_intelligence": {
                "components": [
                    "Self-Awareness",
                    "Self-Regulation",
                    "Motivation",
                    "Empathy",
                    "Social Skills"
                ]
            },
            "competencies": [
                "strategic_thinking",
                "decision_making",
                "communication",
                "team_building",
                "conflict_resolution",
                "adaptability",
                "innovation",
                "empathy",
                "collaboration"
            ]
        }
    
    def analyze_text(self, text: str, company_criteria: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze leadership text using GPT
        
        Args:
            text: Leadership description or interview text
            company_criteria: Optional company-specific requirements
            
        Returns:
            Dict with analysis results
        """
        
        if not self.api_key:
            # Fallback to mock analysis if no API key
            return self._mock_analysis(text)
        
        # Construct prompt
        prompt = self._build_analysis_prompt(text, company_criteria)
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in leadership assessment and organizational psychology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            return self._parse_analysis_response(result_text)
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._mock_analysis(text)
    
    def _build_analysis_prompt(self, text: str, criteria: Optional[Dict]) -> str:
        """Build the analysis prompt"""
        
        prompt = f"""
        Analyze the following leadership description and provide a comprehensive assessment.
        
        Leadership Description:
        {text}
        
        Please evaluate:
        
        1. Leadership Style (rate each 0-10):
           - Transformational Leadership
           - Democratic/Participative
           - Authentic Leadership
           - Servant Leadership
           
        2. Core Competencies (rate each 0-10):
           - Strategic Thinking
           - Decision Making
           - Communication
           - Team Building
           - Conflict Resolution
           - Adaptability
           - Innovation
           - Empathy
           - Collaboration
           
        3. Identify any potential biases or stereotypes in the description
        
        4. Provide 3-5 specific recommendations for development
        
        Format your response as JSON with these keys:
        - leadership_styles: dict of style scores
        - competencies: dict of competency scores
        - biases_detected: list of detected biases
        - recommendations: list of recommendation strings
        """
        
        if criteria:
            prompt += f"\n\nCompany-specific criteria to consider: {json.dumps(criteria)}"
        
        return prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse GPT response into structured format"""
        
        try:
            # Try to parse as JSON
            result = json.loads(response_text)
            
            # Add metadata
            result["timestamp"] = datetime.now().isoformat()
            result["model_used"] = self.model
            
            # Calculate aggregate scores
            result["traditional_score"] = self._calculate_traditional_score(result.get("competencies", {}))
            result["inclusive_score"] = self._calculate_inclusive_score(result.get("competencies", {}))
            
            return result
            
        except json.JSONDecodeError:
            # Fallback parsing if not valid JSON
            return self._fallback_parse(response_text)
    
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
    
    def _calculate_inclusive_score(self, competencies: Dict[str, float]) -> float:
        """Calculate inclusive leadership score"""
        
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
        
        score = sum(
            competencies.get(comp, 5) * weight 
            for comp, weight in inclusive_weights.items()
        )
        
        return min(score, 10)
    
    def _mock_analysis(self, text: str) -> Dict[str, Any]:
        """Provide mock analysis when API is not available"""
        
        # Simple keyword-based analysis for demo
        text_lower = text.lower()
        
        # Count leadership keywords
        empathy_keywords = ["empathy", "understand", "listen", "support", "care"]
        strategic_keywords = ["strategy", "plan", "vision", "goal", "objective"]
        collaborative_keywords = ["team", "together", "collaborate", "share", "collective"]
        
        empathy_score = min(sum(1 for k in empathy_keywords if k in text_lower) * 2, 10)
        strategic_score = min(sum(1 for k in strategic_keywords if k in text_lower) * 2, 10)
        collaborative_score = min(sum(1 for k in collaborative_keywords if k in text_lower) * 2, 10)
        
        competencies = {
            "strategic_thinking": strategic_score,
            "decision_making": 6,
            "communication": 7,
            "team_building": collaborative_score,
            "conflict_resolution": 5,
            "adaptability": 6,
            "innovation": 6,
            "empathy": empathy_score,
            "collaboration": collaborative_score
        }
        
        return {
            "leadership_styles": {
                "transformational": 7,
                "democratic": collaborative_score,
                "authentic": 6,
                "servant": empathy_score
            },
            "competencies": competencies,
            "biases_detected": [],
            "recommendations": [
                "Continue developing your empathetic leadership approach",
                "Consider strategic planning workshops to enhance vision-setting",
                "Practice active listening in team meetings"
            ],
            "traditional_score": self._calculate_traditional_score(competencies),
            "inclusive_score": self._calculate_inclusive_score(competencies),
            "model_used": "mock_analysis",
            "timestamp": datetime.now().isoformat()
        }
    
    def _fallback_parse(self, text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        
        # Basic text analysis to extract some information
        return {
            "raw_analysis": text,
            "parsing_error": "Could not parse structured response",
            "traditional_score": 6.0,
            "inclusive_score": 7.0,
            "recommendations": ["Please review the raw analysis for detailed insights"],
            "timestamp": datetime.now().isoformat()
        }
