# src/multimodal_processor.py
"""
Multimodal Processing Module
Handles audio, video, and questionnaire inputs
"""

import os
import json
import tempfile
from typing import Dict, Optional, Any
from pathlib import Path

# Try importing optional dependencies
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("⚠️ Whisper not installed. Audio processing will be limited.")
    WHISPER_AVAILABLE = False

try:
    import subprocess
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

class MultimodalProcessor:
    """Process various input modalities for leadership assessment"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key
        self.whisper_model = None
        
    def process_audio(self, audio_path: str) -> str:
        """
        Process audio file and extract transcript
        """
        if not audio_path or not os.path.exists(audio_path):
            return ""
            
        try:
            # Option 1: Use OpenAI Whisper API (preferred if API key available)
            if self.openai_api_key:
                import openai
                openai.api_key = self.openai_api_key
                
                with open(audio_path, "rb") as audio_file:
                    transcript = openai.Audio.transcribe("whisper-1", audio_file)
                    return transcript["text"]
            
            # Option 2: Use local Whisper model (fallback)
            elif WHISPER_AVAILABLE:
                if self.whisper_model is None:
                    print("Loading Whisper model... this may take a moment")
                    self.whisper_model = whisper.load_model("base")
                
                result = self.whisper_model.transcribe(audio_path)
                return result["text"]
            else:
                return "[Audio processing unavailable - install whisper or provide OpenAI API key]"
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            return f"[Audio processing failed: {str(e)}]"
    
    def process_video(self, video_path: str) -> str:
        """
        Process video file - extract audio and analyze
        """
        if not video_path or not os.path.exists(video_path):
            return ""
            
        try:
            if FFMPEG_AVAILABLE:
                # Extract audio from video
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_audio_path = temp_audio.name
                    
                # Use ffmpeg to extract audio
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    temp_audio_path, '-y'
                ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Process the extracted audio
                transcript = self.process_audio(temp_audio_path)
                
                # Clean up temp file
                os.unlink(temp_audio_path)
                
                return f"[Video Audio Transcript]: {transcript}"
            else:
                return "[Video processing requires ffmpeg installation]"
                
        except Exception as e:
            print(f"Error processing video: {e}")
            return f"[Video processing failed: {str(e)}]"
    
    def process_questionnaire(self, questionnaire: Dict[str, str]) -> str:
        """
        Process structured questionnaire responses
        """
        if not questionnaire:
            return ""
            
        narrative_parts = []
        
        # Map questions to narrative descriptions
        question_mapping = {
            "leadership_philosophy": "My leadership philosophy is",
            "decision_style": "When making decisions, I",
            "conflict_approach": "In conflict situations, I",
            "team_motivation": "I motivate my team by",
            "change_management": "When managing change, I",
            "communication_style": "My communication style involves",
            "innovation_mindset": "Regarding innovation, I",
            "diversity_commitment": "On diversity and inclusion, I"
        }
        
        for key, value in questionnaire.items():
            if key in question_mapping and value:
                narrative_parts.append(f"{question_mapping[key]} {value}")
            elif value:  # Handle unmapped questions
                narrative_parts.append(f"Regarding {key.replace('_', ' ')}: {value}")
        
        return " ".join(narrative_parts)
    
    def extract_video_features(self, video_path: str) -> Dict[str, Any]:
        """
        Future enhancement: Extract non-verbal cues from video
        Currently returns placeholder
        """
        return {
            "body_language": "confident",
            "eye_contact": "consistent",
            "gestures": "open and inclusive",
            "energy_level": "engaged"
        }
    
    def combine_modalities(self, 
                          text: str = "",
                          audio_transcript: str = "",
                          video_analysis: str = "",
                          questionnaire_text: str = "") -> str:
        """
        Combine all modalities into unified text for analysis
        """
        combined_parts = []
        
        if text:
            combined_parts.append(f"[Direct Input]: {text}")
            
        if audio_transcript:
            combined_parts.append(f"[Audio Content]: {audio_transcript}")
            
        if video_analysis:
            combined_parts.append(f"[Video Analysis]: {video_analysis}")
            
        if questionnaire_text:
            combined_parts.append(f"[Questionnaire Responses]: {questionnaire_text}")
        
        return "\n\n".join(combined_parts) if combined_parts else "No input provided."
