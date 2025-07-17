# src/bias_detector.py
"""
Bias Detection Module for Leadership Assessment
Identifies and corrects gender and neurodiversity biases
"""

import re
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path

class BiasDetector:
    """Detect and correct biases in leadership assessments"""
    
    def __init__(self, bias_patterns_file: Optional[str] = None):
        self.bias_patterns = self._load_bias_patterns(bias_patterns_file)
        self.correction_factors = {
            "double_bind": 0.15,
            "gender_stereotype": 0.10,
            "neurodiversity_bias": 0.12,
            "communication_style": 0.08
        }
    
    def _load_bias_patterns(self, patterns_file: Optional[str]) -> Dict[str, List[str]]:
        """Load bias detection patterns"""
        
        default_patterns = {
            "double_bind": [
                r"too aggressive for a woman",
                r"not assertive enough",
                r"too emotional",
                r"lacks gravitas",
                r"abrasive",
                r"bossy",
                r"needs to be more feminine",
                r"needs to be more authoritative"
            ],
            "gender_stereotype": [
                r"natural caregiver",
                r"maternal approach",
                r"nurturing style",
                r"soft skills only",
                r"better with people than strategy",
                r"not technical enough",
                r"too nice to lead"
            ],
            "neurodiversity_bias": [
                r"lacks social awareness",
                r"too focused on details",
                r"inflexible",
                r"poor eye contact",
                r"awkward in meetings",
                r"obsessive about process",
                r"doesn't read the room"
            ],
            "communication_style": [
                r"talks too much",
                r"doesn't speak up enough",
                r"overly collaborative",
                r"indecisive",
                r"takes too long to decide"
            ]
        }
        
        if patterns_file and Path(patterns_file).exists():
            with open(patterns_file, 'r') as f:
                loaded_patterns = json.load(f)
                default_patterns.update(loaded_patterns)
        
        return default_patterns
    
    def detect_biases(self, text: str) -> Dict[str, any]:
        """Detect biases in assessment text"""
        
        detected_biases = []
        bias_details = {}
        
        text_lower = text.lower()
        
        for bias_type, patterns in self.bias_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    matches.append(pattern)
            
            if matches:
                detected_biases.append(bias_type)
                bias_details[bias_type] = {
                    "patterns_found": matches,
                    "severity": len(matches) / len(patterns),
                    "correction_factor": self.correction_factors.get(bias_type, 0.1)
                }
        
        return {
            "biases": detected_biases,
            "details": bias_details,
            "corrections": self._calculate_corrections(bias_details),
            "bias_free": len(detected_biases) == 0
        }
    
    def _calculate_corrections(self, bias_details: Dict) -> Dict[str, float]:
        """Calculate score corrections for detected biases"""
        
        corrections = {}
        
        # Competencies that are often undervalued due to bias
        affected_competencies = {
            "double_bind": ["leadership_presence", "assertiveness", "strategic_thinking"],
            "gender_stereotype": ["technical_skills", "analytical_thinking", "innovation"],
            "neurodiversity_bias": ["attention_to_detail", "systematic_thinking", "quality_focus"],
            "communication_style": ["collaboration", "consensus_building", "team_harmony"]
        }
        
        for bias_type, details in bias_details.items():
            severity = details["severity"]
            correction = details["correction_factor"] * severity
            
            for competency in affected_competencies.get(bias_type, []):
                corrections[competency] = corrections.get(competency, 0) + correction
        
        return corrections
    
    def apply_corrections(self, scores: Dict[str, float], corrections: Dict[str, float]) -> Dict[str, float]:
        """Apply bias corrections to competency scores"""
        
        corrected_scores = scores.copy()
        
        for competency, correction in corrections.items():
            if competency in corrected_scores:
                # Increase score by correction factor, max 10
                corrected_scores[competency] = min(
                    corrected_scores[competency] * (1 + correction),
                    10.0
                )
        
        return corrected_scores
    
    def generate_bias_report(self, bias_analysis: Dict) -> str:
        """Generate human-readable bias report"""
        
        if bias_analysis["bias_free"]:
            return "✅ No significant biases detected in this assessment."
        
        report = "⚠️ Bias Detection Report:\n\n"
        
        for bias_type, details in bias_analysis["details"].items():
            report += f"**{bias_type.replace('_', ' ').title()}**\n"
            report += f"- Severity: {details['severity']:.0%}\n"
            report += f"- Patterns found: {len(details['patterns_found'])}\n"
            report += f"- Correction applied: +{details['correction_factor']*100:.0f}%\n\n"
        
        report += "\n💡 The inclusive score has been adjusted to compensate for these biases."
        
        return report
