# src/ml_enhanced_integration.py
"""
Enhanced ML Integration with Multiple Models and Data Collection
Supports OpenAI models (with free tier) and open-source alternatives
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib

# Model availability tracking
@dataclass
class ModelUsageTracker:
    """Track model usage and availability"""
    model_name: str
    daily_limit: int
    tokens_used: int = 0
    last_reset: datetime = None
    
    def can_use(self, estimated_tokens: int = 1000) -> bool:
        """Check if model can be used"""
        # Reset daily counter
        if self.last_reset is None or \
           (datetime.now() - self.last_reset).days >= 1:
            self.tokens_used = 0
            self.last_reset = datetime.now()
        
        return (self.tokens_used + estimated_tokens) <= self.daily_limit
    
    def update_usage(self, tokens: int):
        """Update token usage"""
        self.tokens_used += tokens

class MultiModelManager:
    """Manage multiple ML models with fallback logic"""
    
    # Model limits based on your access
    MODEL_LIMITS = {
        "gpt-4.5-preview": 250_000,
        "gpt-4o": 250_000,
        "o1": 250_000,
        "gpt-4o-mini": 2_500_000,
        "o1-mini": 2_500_000,
        "o3-mini": 2_500_000,
        "llama2-7b": float('inf'),  # No limit for local models
        "mistral-7b": float('inf')
    }
    
    # Model priorities (best to fallback)
    MODEL_PRIORITY = [
        "gpt-4o",           # Best for complex analysis
        "gpt-4.5-preview",  # Good alternative
        "o1",               # Another option
        "gpt-4o-mini",      # Lighter but capable
        "o1-mini",          # Fallback
        "o3-mini",          # Last OpenAI option
        "llama2-7b",        # Open source fallback
        "mistral-7b"        # Final fallback
    ]
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.usage_trackers = {
            model: ModelUsageTracker(model, limit)
            for model, limit in self.MODEL_LIMITS.items()
        }
        self.load_usage_state()
        
    def select_model(self, task_complexity: str = "medium") -> Tuple[str, str]:
        """
        Select best available model based on task and usage limits
        Returns: (model_name, model_type)
        """
        # Estimate tokens based on complexity
        token_estimates = {
            "simple": 500,
            "medium": 1500,
            "complex": 3000
        }
        estimated_tokens = token_estimates.get(task_complexity, 1500)
        
        # Try models in priority order
        for model in self.MODEL_PRIORITY:
            if model in self.usage_trackers:
                tracker = self.usage_trackers[model]
                if tracker.can_use(estimated_tokens):
                    # Determine model type
                    if model.startswith(("gpt", "o1", "o3")):
                        model_type = "openai"
                    else:
                        model_type = "opensource"
                    
                    return model, model_type
        
        # If all limits exhausted, use open source
        return "llama2-7b", "opensource"
    
    def record_usage(self, model_name: str, tokens_used: int):
        """Record token usage for a model"""
        if model_name in self.usage_trackers:
            self.usage_trackers[model_name].update_usage(tokens_used)
            self.save_usage_state()
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get current usage statistics"""
        stats = {}
        for model, tracker in self.usage_trackers.items():
            stats[model] = {
                "daily_limit": tracker.daily_limit,
                "tokens_used": tracker.tokens_used,
                "tokens_remaining": tracker.daily_limit - tracker.tokens_used,
                "percentage_used": (tracker.tokens_used / tracker.daily_limit * 100) 
                                  if tracker.daily_limit != float('inf') else 0
            }
        return stats
    
    def save_usage_state(self):
        """Save usage state to file"""
        state_file = Path("data/model_usage_state.json")
        state_file.parent.mkdir(exist_ok=True)
        
        state = {}
        for model, tracker in self.usage_trackers.items():
            state[model] = {
                "tokens_used": tracker.tokens_used,
                "last_reset": tracker.last_reset.isoformat() if tracker.last_reset else None
            }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_usage_state(self):
        """Load usage state from file"""
        state_file = Path("data/model_usage_state.json")
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            for model, data in state.items():
                if model in self.usage_trackers:
                    self.usage_trackers[model].tokens_used = data["tokens_used"]
                    if data["last_reset"]:
                        self.usage_trackers[model].last_reset = \
                            datetime.fromisoformat(data["last_reset"])

# Data collection for future ML training
@dataclass
class AssessmentData:
    """Store assessment data for ML training"""
    id: str
    timestamp: datetime
    model_used: str
    input_text: str
    
    # Analysis results
    leadership_scores: Dict[str, float]
    competencies: Dict[str, float]
    bias_detected: List[str]
    
    # User feedback (for reinforcement learning)
    user_rating: Optional[int] = None  # 1-5 stars
    user_feedback: Optional[str] = None
    corrections: Optional[Dict[str, Any]] = None
    
    def to_training_format(self) -> Dict[str, Any]:
        """Convert to format suitable for ML training"""
        return {
            "input": self.input_text,
            "output": {
                "leadership_scores": self.leadership_scores,
                "competencies": self.competencies,
                "bias_detected": self.bias_detected
            },
            "metadata": {
                "model": self.model_used,
                "rating": self.user_rating,
                "has_corrections": bool(self.corrections)
            }
        }

class DataCollector:
    """Collect and manage training data"""
    
    def __init__(self, data_dir: str = "data/assessments"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def save_assessment(self, assessment: AssessmentData):
        """Save assessment data"""
        filename = f"{assessment.id}_{assessment.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            data = asdict(assessment)
            # Convert datetime to string
            data['timestamp'] = data['timestamp'].isoformat()
            json.dump(data, f, indent=2)
    
    def load_all_assessments(self) -> List[AssessmentData]:
        """Load all saved assessments"""
        assessments = []
        
        for filepath in self.data_dir.glob("*.json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Convert string back to datetime
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                assessments.append(AssessmentData(**data))
        
        return assessments
    
    def get_training_dataset(self, min_rating: int = 4) -> pd.DataFrame:
        """Get high-quality data for training"""
        assessments = self.load_all_assessments()
        
        # Filter high-quality assessments
        quality_data = [
            a.to_training_format() 
            for a in assessments 
            if a.user_rating and a.user_rating >= min_rating
        ]
        
        return pd.DataFrame(quality_data)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics"""
        assessments = self.load_all_assessments()
        
        if not assessments:
            return {
                "total_assessments": 0,
                "with_feedback": 0,
                "average_rating": 0,
                "models_used": {}
            }
        
        # Calculate statistics
        total = len(assessments)
        with_feedback = sum(1 for a in assessments if a.user_feedback)
        ratings = [a.user_rating for a in assessments if a.user_rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Model usage
        model_counts = {}
        for a in assessments:
            model_counts[a.model_used] = model_counts.get(a.model_used, 0) + 1
        
        return {
            "total_assessments": total,
            "with_feedback": with_feedback,
            "average_rating": round(avg_rating, 2),
            "models_used": model_counts,
            "ready_for_training": len([a for a in assessments 
                                     if a.user_rating and a.user_rating >= 4])
        }

# ML Training Pipeline (Simplified for MVP)
class SimpleMLTrainer:
    """Simple ML training pipeline for leadership patterns"""
    
    def __init__(self, data_collector: DataCollector):
        self.data_collector = data_collector
        self.model_version = "v0.1"
        self.training_history = []
        
    def can_train(self) -> Tuple[bool, str]:
        """Check if we have enough data to train"""
        stats = self.data_collector.get_statistics()
        ready_count = stats["ready_for_training"]
        
        if ready_count < 10:
            return False, f"Need at least 10 high-quality assessments (have {ready_count})"
        elif ready_count < 50:
            return True, f"Can do basic training with {ready_count} assessments"
        else:
            return True, f"Ready for full training with {ready_count} assessments"
    
    def train_bias_detector(self) -> Dict[str, Any]:
        """Train custom bias detection patterns"""
        # For MVP: Simple pattern learning
        assessments = self.data_collector.load_all_assessments()
        
        # Extract bias patterns from high-rated assessments
        bias_patterns = {}
        for a in assessments:
            if a.user_rating and a.user_rating >= 4:
                for bias in a.bias_detected:
                    bias_patterns[bias] = bias_patterns.get(bias, 0) + 1
        
        # Save learned patterns
        patterns_file = Path("models/learned_bias_patterns.json")
        patterns_file.parent.mkdir(exist_ok=True)
        
        with open(patterns_file, 'w') as f:
            json.dump(bias_patterns, f, indent=2)
        
        return {
            "patterns_learned": len(bias_patterns),
            "most_common": sorted(bias_patterns.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
        }
    
    def simulate_training(self) -> Dict[str, Any]:
        """Simulate ML training for demo purposes"""
        stats = self.data_collector.get_statistics()
        
        # Simulate training metrics
        training_result = {
            "timestamp": datetime.now().isoformat(),
            "model_version": self.model_version,
            "data_used": stats["ready_for_training"],
            "metrics": {
                "accuracy": min(0.75 + (stats["ready_for_training"] / 200), 0.95),
                "bias_detection_f1": min(0.70 + (stats["ready_for_training"] / 250), 0.92),
                "loss": max(0.45 - (stats["ready_for_training"] / 300), 0.15)
            },
            "improvements": [
                "Better detection of double-bind patterns",
                "Improved neurodiversity awareness",
                "Enhanced context understanding"
            ]
        }
        
        self.training_history.append(training_result)
        return training_result

# Integration with main app
class EnhancedLeadershipAnalyzer:
    """Enhanced analyzer with ML capabilities"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.model_manager = MultiModelManager(openai_api_key)
        self.data_collector = DataCollector()
        self.ml_trainer = SimpleMLTrainer(self.data_collector)
        
    def analyze_with_ml(self, 
                       text: str, 
                       task_complexity: str = "medium",
                       save_for_training: bool = True) -> Dict[str, Any]:
        """Analyze leadership with ML model selection and data collection"""
        
        # Select best available model
        model_name, model_type = self.model_manager.select_model(task_complexity)
        
        # Generate assessment ID
        assessment_id = hashlib.md5(
            f"{text[:50]}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:8]
        
        # Perform analysis (simplified for demo)
        # In real implementation, this would call actual models
        analysis_result = {
            "model_used": model_name,
            "model_type": model_type,
            "leadership_scores": {
                "transformational": np.random.uniform(6, 9),
                "democratic": np.random.uniform(5, 8),
                "inclusive": np.random.uniform(7, 10)
            },
            "competencies": {
                "strategic_thinking": np.random.uniform(5, 9),
                "empathy": np.random.uniform(6, 10),
                "innovation": np.random.uniform(5, 9),
                "communication": np.random.uniform(6, 9)
            },
            "bias_detected": [],
            "ml_confidence": 0.85
        }
        
        # Record token usage (estimated)
        tokens_used = len(text.split()) * 3  # Rough estimate
        self.model_manager.record_usage(model_name, tokens_used)
        
        # Save for training if requested
        if save_for_training:
            assessment = AssessmentData(
                id=assessment_id,
                timestamp=datetime.now(),
                model_used=model_name,
                input_text=text,
                leadership_scores=analysis_result["leadership_scores"],
                competencies=analysis_result["competencies"],
                bias_detected=analysis_result["bias_detected"]
            )
            self.data_collector.save_assessment(assessment)
        
        # Add ML statistics
        analysis_result["ml_stats"] = {
            "data_collected": self.data_collector.get_statistics(),
            "model_usage": self.model_manager.get_usage_stats(),
            "can_train": self.ml_trainer.can_train()
        }
        
        return analysis_result
    
    def add_user_feedback(self, assessment_id: str, rating: int, feedback: str):
        """Add user feedback to assessment"""
        # In real implementation, would update the saved assessment
        print(f"Feedback recorded for {assessment_id}: {rating} stars")
    
    def trigger_training(self) -> Dict[str, Any]:
        """Trigger ML model training"""
        can_train, message = self.ml_trainer.can_train()
        
        if not can_train:
            return {"success": False, "message": message}
        
        # Train bias detector
        bias_results = self.ml_trainer.train_bias_detector()
        
        # Simulate full training
        training_results = self.ml_trainer.simulate_training()
        
        return {
            "success": True,
            "message": "Training completed successfully",
            "bias_detection": bias_results,
            "training_metrics": training_results
        }

# Usage example for Gradio integration
if __name__ == "__main__":
    # Initialize enhanced analyzer
    analyzer = EnhancedLeadershipAnalyzer()
    
    # Example analysis
    result = analyzer.analyze_with_ml(
        text="I believe in empowering my team through transparent communication...",
        task_complexity="medium"
    )
    
    print(f"Analysis completed using: {result['model_used']}")
    print(f"ML Statistics: {result['ml_stats']}")
    
    # Check if we can train
    can_train, message = analyzer.ml_trainer.can_train()
    print(f"Training status: {message}")
