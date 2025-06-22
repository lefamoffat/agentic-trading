#!/usr/bin/env python3
"""LLM advisor for intelligent trading configuration recommendations.

This module uses Large Language Models to analyze trading experiment data
and provide intelligent recommendations for configuration optimization.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

class LLMAdvisor:
    """Provides LLM-powered recommendations for trading experiments."""
    
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        """Initialize LLM advisor.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (defaults to environment variable)
            
        Raises:
            ConfigurationError: If OpenAI API key is not available
        """
        self.logger = get_logger(self.__class__.__name__)
        self.model = model
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def get_recommendations(
        self,
        action: str,
        experiment_data: Dict[str, Any],
        symbol: str = "EUR/USD",
        **kwargs
    ) -> Dict[str, Any]:
        """Get LLM recommendations for trading experiment configuration.
        
        Args:
            action: Type of action (start, continue, optimize)
            experiment_data: Analyzed experiment data
            symbol: Trading symbol
            **kwargs: Additional parameters
            
        Returns:
            LLM recommendations with reasoning
            
        Raises:
            ConfigurationError: If LLM request fails
        """
        try:
            # Create prompt based on action type
            prompt = self._create_prompt(action, experiment_data, symbol)
            
            # Query OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert trading AI advisor. Provide detailed, data-driven recommendations in JSON format."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            
            # Parse response
            recommendations = self._parse_llm_response(response.choices[0].message.content)
            return recommendations
            
        except Exception as e:
            self.logger.error(f"LLM recommendation failed: {e}")
            raise ConfigurationError(
                f"Failed to get LLM recommendations for {action} action: {e}"
            ) from e
    
    def _create_prompt(self, action: str, experiment_data: Dict[str, Any], symbol: str) -> str:
        """Create prompt for LLM based on action type."""
        base_context = f"""
You are analyzing trading experiments for {symbol}. Based on the following data, provide recommendations in JSON format.

Experiment Data:
{json.dumps(experiment_data, indent=2)}

Action: {action}
"""
        
        if action == "start":
            return base_context + """
Task: Recommend configuration for a new trading experiment based on historical performance.

Provide JSON response with:
{
    "reasoning": "Your analysis and rationale",
    "confidence_score": 0.85,
    "recommended_changes": {
        "learning_rate": {"value": 0.0003, "reason": "explanation"},
        "initial_balance": {"value": 100000, "reason": "explanation"}
    }
}
"""
        
        elif action == "continue":
            return base_context + """
Task: Analyze current experiment and recommend whether to continue, modify, or restart.

Provide JSON response with:
{
    "reasoning": "Your analysis of current performance",
    "confidence_score": 0.85,
    "action_recommendation": "continue|modify|restart",
    "recommended_changes": {
        "learning_rate": {"current": 0.0003, "new": 0.0001, "reason": "explanation"}
    }
}
"""
        
        else:  # optimize
            return base_context + """
Task: Suggest optimizations for better performance.

Provide JSON response with:
{
    "reasoning": "Your optimization analysis",
    "confidence_score": 0.85,
    "optimization_strategy": "description",
    "recommended_changes": {
        "parameter": {"value": "new_value", "reason": "explanation"}
    }
}
"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse and validate LLM response.
        
        Raises:
            ConfigurationError: If response cannot be parsed
        """
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                recommendations = json.loads(json_str)
                
                # Validate confidence score
                confidence = recommendations.get("confidence_score", 0.5)
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    recommendations["confidence_score"] = 0.5
                
                return recommendations
            else:
                raise ValueError("No valid JSON found in response")
                
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise ConfigurationError(
                f"Failed to parse LLM response: {e}"
            ) from e 