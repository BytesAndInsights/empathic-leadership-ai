# src/bias_detector.py
"""
Bias Detection Module
Identifies and corrects gender and neurodiversity biases in leadership assessment
"""

import re
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

class BiasDetector:
    """Detect and correct biases in leadership assessment"""
    
    def __init__(self, patterns_file: Optional[str] = None):
        self.bias_patterns = self._load_bias_patterns(patterns_file)
        self.correction_factors = self._initialize_correction_factors()
        
    def _load_bias_patterns(self, patterns_file: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Load bias detection patterns"""
        
        default_patterns = {
            "double_bind": [
                {
                    "pattern": r"too (aggressive|assertive|pushy) for a (woman|female)",
                    "description": "Penalizing assertiveness in women leaders"
                },
                {
                    "pattern": r"(not assertive|too soft|weak) (enough )?for (leadership|a leader)",
                    "description": "Requiring traditionally masculine traits"
                },
                {
                    "pattern": r"(emotional|too emotional) (rather than|instead of) (logical|rational)",
                    "description": "Dismissing emotional intelligence as weakness"
                }
            ],
            "gender_stereotypes": [
                {
                    "pattern": r"natural (caregiver|nurturer|mother figure)",
                    "description": "Reducing leadership to caregiving roles"
                },
                {
                    "pattern": r"(lacks|doesn't have) technical (skills|expertise|knowledge)",
                    "description": "Assuming technical incompetence"
                },
                {
                    "pattern": r"better with (people|soft skills) than (strategy|hard skills)",
                    "description": "Limiting to interpersonal roles only"
                }
            ],
            "neurodiversity_bias": [
                {
                    "pattern": r"(lacks|poor|weak) social (skills|awareness|intelligence)",
                    "description": "Penalizing different social interaction styles"
                },
                {
                    "pattern": r"too (focused on|obsessed with) details",
                    "description": "Dismissing thoroughness as obsession"
                },
                {
                    "pattern": r"(inflexible|rigid) thinking",
                    "description": "Not recognizing systematic thinking strengths"
                }
            ],
            "communication_style_bias": [
                {
                    "pattern": r"(talks too much|overly talkative|verbose)",
                    "description": "Penalizing collaborative communication"
                },
                {
                    "pattern": r"(too quiet|doesn't speak up|passive)",
                    "description": "Requiring extroverted communication style"
                }
            ]
        }
        
        if patterns_file and Path(patterns_file).exists():
            with open(patterns_file, 'r') as f:
                custom_patterns = json.load(f)
                default_patterns.update(custom_patterns)
                
        return default_patterns
    
    def _initialize_correction_factors(self) -> Dict[str, float]:
        """Initialize bias correction factors for competencies"""
        return {
            "double_bind": {
                "assertiveness": 0.15,
                "decision_making": 0.10,
                "strategic_thinking": 0.10
            },
            "gender_stereotypes": {
                "technical_competence": 0.20,
                "innovation": 0.15,
                "strategic_thinking": 0.15
            },
            "neurodiversity_bias": {
                "communication": 0.10,
                "collaboration": 0.10,
                "adaptability": 0.15
            },
            "communication_style_bias": {
                "communication": 0.15,
                "team_building": 0.10
            }
        }
    
    def detect_biases(self, text: str) -> Dict[str, Any]:
        """
        Detect biases in assessment text
        Returns detected biases and suggested corrections
        """
        detected_biases = []
        bias_details = []
        corrections = {}
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # Check each bias category
        for bias_type, patterns in self.bias_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                if re.search(pattern, text_lower):
                    detected_biases.append(bias_type)
                    bias_details.append({
                        "type": bias_type,
                        "pattern_matched": pattern,
                        "description": pattern_info["description"]
                    })
                    
                    # Add corrections for this bias type
                    if bias_type in self.correction_factors:
                        for competency, factor in self.correction_factors[bias_type].items():
                            if competency not in corrections:
                                corrections[competency] = 0
                            corrections[competency] += factor
        
        # Remove duplicates
        detected_biases = list(set(detected_biases))
        
        return {
            "biases": detected_biases,
            "details": bias_details,
            "corrections": corrections,
            "bias_score": len(bias_details),  # Simple count for now
            "recommendations": self._generate_recommendations(detected_biases)
        }
    
    def correct_scores(self, 
                      original_scores: Dict[str, float], 
                      detected_biases: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply bias corrections to competency scores
        """
        corrected_scores = original_scores.copy()
        corrections = detected_biases.get("corrections", {})
        
        for competency, correction_factor in corrections.items():
            if competency in corrected_scores:
                # Apply correction (increase score to compensate for bias)
                original = corrected_scores[competency]
                corrected = min(10, original * (1 + correction_factor))
                corrected_scores[competency] = corrected
        
        return corrected_scores
    
    def _generate_recommendations(self, detected_biases: List[str]) -> List[str]:
        """Generate recommendations based on detected biases"""
        recommendations = []
        
        if "double_bind" in detected_biases:
            recommendations.append(
                "Consider evaluating assertiveness in context rather than against gender norms"
            )
            
        if "gender_stereotypes" in detected_biases:
            recommendations.append(
                "Focus on actual competencies demonstrated rather than stereotypical expectations"
            )
            
        if "neurodiversity_bias" in detected_biases:
            recommendations.append(
                "Recognize diverse communication and thinking styles as potential strengths"
            )
            
        if "communication_style_bias" in detected_biases:
            recommendations.append(
                "Value different communication approaches rather than requiring one style"
            )
        
        if not recommendations:
            recommendations.append(
                "No significant biases detected. Continue with inclusive assessment practices."
            )
            
        return recommendations
    
    def analyze_language_bias(self, text: str) -> Dict[str, Any]:
        """
        Analyze language for subtle biases
        """
        # Word frequency analysis for gendered language
        masculine_words = ["aggressive", "dominant", "competitive", "decisive", "ambitious"]
        feminine_words = ["collaborative", "nurturing", "supportive", "empathetic", "caring"]
        
        text_lower = text.lower()
        masculine_count = sum(1 for word in masculine_words if word in text_lower)
        feminine_count = sum(1 for word in feminine_words if word in text_lower)
        
        return {
            "masculine_language_ratio": masculine_count / (masculine_count + feminine_count + 1),
            "feminine_language_ratio": feminine_count / (masculine_count + feminine_count + 1),
            "language_balance": abs(masculine_count - feminine_count) < 3
        }
    
    def get_bias_report(self, text: str, scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate comprehensive bias report
        """
        # Detect biases
        bias_analysis = self.detect_biases(text)
        
        # Analyze language
        language_analysis = self.analyze_language_bias(text)
        
        # Correct scores
        corrected_scores = self.correct_scores(scores, bias_analysis)
        
        # Calculate impact
        total_correction = sum(
            corrected_scores[k] - scores[k] 
            for k in scores 
            if k in corrected_scores
        )
        
        return {
            "biases_detected": bias_analysis["biases"],
            "bias_details": bias_analysis["details"],
            "language_analysis": language_analysis,
            "original_scores": scores,
            "corrected_scores": corrected_scores,
            "total_correction_impact": total_correction,
            "recommendations": bias_analysis["recommendations"]
        }
