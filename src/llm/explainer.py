"""
LLM-based explanation generation
Supports OpenAI API and local LLMs
"""

import os
from typing import Optional, Dict, List
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMExplainer:
    """
    Generate natural language explanations using LLMs
    
    Args:
        provider: LLM provider ('openai' or 'local')
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        api_key: API key for OpenAI (optional, uses env var)
        temperature: Sampling temperature
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        
        if provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        elif provider == "local":
            # TODO: Implement local LLM support (Llama, etc.)
            logger.warning("Local LLM not yet implemented")
            self.client = None
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def create_classification_prompt(
        self,
        predicted_class: str,
        confidence: float,
        top_k_classes: List[tuple],
        attribution_summary: str,
        concepts: Optional[List[str]] = None,
        uncertainty: Optional[float] = None
    ) -> str:
        """
        Create prompt for classification explanation
        
        Args:
            predicted_class: Predicted class name
            confidence: Prediction confidence
            top_k_classes: List of (class, probability) tuples
            attribution_summary: Summary of attribution analysis
            concepts: List of detected concepts
            uncertainty: Uncertainty estimate
            
        Returns:
            prompt: Formatted prompt
        """
        prompt = f"""You are an AI explainability assistant. Explain the following model prediction to a domain expert.

**Prediction:**
- Class: {predicted_class}
- Confidence: {confidence:.2%}

**Top Predictions:**
"""
        for cls, prob in top_k_classes[:3]:
            prompt += f"- {cls}: {prob:.2%}\n"
        
        prompt += f"\n**Attribution Analysis:**\n{attribution_summary}\n"
        
        if concepts:
            prompt += f"\n**Detected Concepts:**\n"
            for concept in concepts:
                prompt += f"- {concept}\n"
        
        if uncertainty is not None:
            prompt += f"\n**Uncertainty:** {uncertainty:.3f}\n"
        
        prompt += """
Please provide:
1. A clear explanation of why the model made this prediction
2. Key visual features that influenced the decision
3. Confidence assessment and potential caveats
4. Actionable insights for the user

Keep the explanation concise and accessible.
"""
        
        return prompt
    
    def create_segmentation_prompt(
        self,
        class_distribution: Dict[str, float],
        attribution_summary: str,
        iou_score: Optional[float] = None,
        uncertainty: Optional[float] = None
    ) -> str:
        """
        Create prompt for segmentation explanation
        
        Args:
            class_distribution: Dict of {class: pixel_percentage}
            attribution_summary: Summary of attribution analysis
            iou_score: IoU score if ground truth available
            uncertainty: Uncertainty estimate
            
        Returns:
            prompt: Formatted prompt
        """
        prompt = f"""You are an AI explainability assistant. Explain the following segmentation result to a domain expert.

**Segmentation Result:**
"""
        for cls, pct in class_distribution.items():
            prompt += f"- {cls}: {pct:.1%} of image\n"
        
        prompt += f"\n**Attribution Analysis:**\n{attribution_summary}\n"
        
        if iou_score is not None:
            prompt += f"\n**IoU Score:** {iou_score:.3f}\n"
        
        if uncertainty is not None:
            prompt += f"\n**Uncertainty:** {uncertainty:.3f}\n"
        
        prompt += """
Please provide:
1. An explanation of the segmentation quality
2. Regions where the model is most/least confident
3. Potential improvements or failure modes
4. Clinical or practical implications (if applicable)

Keep the explanation concise and actionable.
"""
        
        return prompt
    
    def generate_explanation(
        self,
        prompt: str,
        max_tokens: int = 500
    ) -> str:
        """
        Generate explanation using LLM
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            explanation: Generated text explanation
        """
        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an AI model explainability expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                
                explanation = response.choices[0].message.content
                
                logger.info(f"Generated explanation ({len(explanation)} chars)")
                
                return explanation
            
            except Exception as e:
                logger.error(f"Failed to generate explanation: {e}")
                return f"Error generating explanation: {str(e)}"
        
        elif self.provider == "local":
            # TODO: Implement local LLM inference
            return "Local LLM not yet implemented"
        
        else:
            return "Unknown provider"
    
    def explain_classification(
        self,
        predicted_class: str,
        confidence: float,
        top_k_classes: List[tuple],
        attribution_summary: str,
        concepts: Optional[List[str]] = None,
        uncertainty: Optional[float] = None
    ) -> str:
        """
        Generate classification explanation (convenience method)
        """
        prompt = self.create_classification_prompt(
            predicted_class, confidence, top_k_classes,
            attribution_summary, concepts, uncertainty
        )
        return self.generate_explanation(prompt)
    
    def explain_segmentation(
        self,
        class_distribution: Dict[str, float],
        attribution_summary: str,
        iou_score: Optional[float] = None,
        uncertainty: Optional[float] = None
    ) -> str:
        """
        Generate segmentation explanation (convenience method)
        """
        prompt = self.create_segmentation_prompt(
            class_distribution, attribution_summary, iou_score, uncertainty
        )
        return self.generate_explanation(prompt)


# TODO: Add prompt templates library
# TODO: Add few-shot learning examples
# TODO: Add multi-modal LLM support (vision + text)
# TODO: Add local LLM support (Llama, Mistral)
# TODO: Add explanation quality metrics
