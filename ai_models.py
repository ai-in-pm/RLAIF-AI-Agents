import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

import openai
import anthropic
import google.generativeai as genai

from utils import load_api_keys

logger = logging.getLogger(__name__)

class AIModel(ABC):
    """Abstract base class for AI models used in RLAIF."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        pass
    
    @abstractmethod
    def compare(self, prompt: str, response_a: str, response_b: str, **kwargs) -> Dict[str, float]:
        """Compare two responses and return preference scores."""
        pass

class OpenAIModel(AIModel):
    """OpenAI model wrapper."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize the OpenAI model.
        
        Args:
            model_name: The name of the OpenAI model to use
            api_key: Optional API key. If not provided, will use environment variable
        """
        self.model_name = model_name
        
        # Set up the API key
        if api_key:
            openai.api_key = api_key
        else:
            # Use environment variable
            api_keys = load_api_keys()
            openai.api_key = api_keys['openai']
            
        logger.info(f"Initialized OpenAI model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the OpenAI model."""
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating response from OpenAI model: {e}")
            return ""
    
    def compare(self, prompt: str, response_a: str, response_b: str, **kwargs) -> Dict[str, float]:
        """Compare two responses using the OpenAI model."""
        comparison_prompt = f"""Given the following prompt and two responses, evaluate which response is better and why.
        
        Prompt: {prompt}
        
        Response A:
        {response_a}
        
        Response B:
        {response_b}
        
        First provide your analysis, then rate each response on a scale of 0-10 for helpfulness, accuracy, and safety.
        Finally, indicate which response (A or B) is better overall.
        """
        
        try:
            analysis = self.generate(comparison_prompt, **kwargs)
            
            # For simplicity, extract scores using another API call
            score_prompt = f"""Based on the following analysis of two responses A and B, provide a JSON containing scores.
            
            Analysis:
            {analysis}
            
            Return a JSON object with the following format:
            {{
                "response_a": {{
                    "helpfulness": <score from 0-10>,
                    "accuracy": <score from 0-10>,
                    "safety": <score from 0-10>
                }},
                "response_b": {{
                    "helpfulness": <score from 0-10>,
                    "accuracy": <score from 0-10>,
                    "safety": <score from 0-10>
                }},
                "preferred": "A or B"
            }}
            """
            
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": score_prompt}],
                response_format={"type": "json_object"},
                **kwargs
            )
            
            import json
            scores = json.loads(response.choices[0].message.content)
            return scores
            
        except Exception as e:
            logger.error(f"Error comparing responses with OpenAI model: {e}")
            return {
                "response_a": {"helpfulness": 5, "accuracy": 5, "safety": 5},
                "response_b": {"helpfulness": 5, "accuracy": 5, "safety": 5},
                "preferred": "A"
            }

class AnthropicModel(AIModel):
    """Anthropic model wrapper."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize the Anthropic model.
        
        Args:
            model_name: The name of the Anthropic model to use
            api_key: Optional API key. If not provided, will use environment variable
        """
        self.model_name = model_name
        
        # Set up the API key
        if api_key:
            self.api_key = api_key
        else:
            # Use environment variable
            api_keys = load_api_keys()
            self.api_key = api_keys['anthropic']
            
        self.client = anthropic.Anthropic(api_key=self.api_key)
        logger.info(f"Initialized Anthropic model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the Anthropic model."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating response from Anthropic model: {e}")
            return ""
    
    def compare(self, prompt: str, response_a: str, response_b: str, **kwargs) -> Dict[str, float]:
        """Compare two responses using the Anthropic model."""
        comparison_prompt = f"""Given the following prompt and two responses, evaluate which response is better and why.
        
        Prompt: {prompt}
        
        Response A:
        {response_a}
        
        Response B:
        {response_b}
        
        First provide your analysis, then rate each response on a scale of 0-10 for helpfulness, accuracy, and safety.
        Finally, indicate which response (A or B) is better overall.
        """
        
        try:
            analysis = self.generate(comparison_prompt, **kwargs)
            
            # For simplicity, extract scores using another API call
            score_prompt = f"""Based on the following analysis of two responses A and B, provide a JSON containing scores.
            
            Analysis:
            {analysis}
            
            Return a JSON object with the following format:
            {{
                "response_a": {{
                    "helpfulness": <score from 0-10>,
                    "accuracy": <score from 0-10>,
                    "safety": <score from 0-10>
                }},
                "response_b": {{
                    "helpfulness": <score from 0-10>,
                    "accuracy": <score from 0-10>,
                    "safety": <score from 0-10>
                }},
                "preferred": "A or B"
            }}
            """
            
            json_response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": score_prompt}],
                max_tokens=1024,
                **kwargs
            )
            
            import json
            import re
            
            # Extract JSON from the response
            response_text = json_response.content[0].text
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)
                return scores
            else:
                raise ValueError("JSON not found in response")
            
        except Exception as e:
            logger.error(f"Error comparing responses with Anthropic model: {e}")
            return {
                "response_a": {"helpfulness": 5, "accuracy": 5, "safety": 5},
                "response_b": {"helpfulness": 5, "accuracy": 5, "safety": 5},
                "preferred": "A"
            }

class GoogleModel(AIModel):
    """Google Generative AI model wrapper."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """Initialize the Google model.
        
        Args:
            model_name: The name of the Google model to use
            api_key: Optional API key. If not provided, will use environment variable
        """
        self.model_name = model_name
        
        # Set up the API key
        if api_key:
            self.api_key = api_key
        else:
            # Use environment variable
            api_keys = load_api_keys()
            self.api_key = api_keys['google']
            
        # Configure the generative AI client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)
        logger.info(f"Initialized Google model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the Google model."""
        try:
            response = self.model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response from Google model: {e}")
            return ""
    
    def compare(self, prompt: str, response_a: str, response_b: str, **kwargs) -> Dict[str, float]:
        """Compare two responses using the Google model."""
        comparison_prompt = f"""Given the following prompt and two responses, evaluate which response is better and why.
        
        Prompt: {prompt}
        
        Response A:
        {response_a}
        
        Response B:
        {response_b}
        
        First provide your analysis, then rate each response on a scale of 0-10 for helpfulness, accuracy, and safety.
        Finally, indicate which response (A or B) is better overall.
        """
        
        try:
            analysis = self.generate(comparison_prompt, **kwargs)
            
            # For simplicity, extract scores using another API call
            score_prompt = f"""Based on the following analysis of two responses A and B, provide a JSON containing scores.
            
            Analysis:
            {analysis}
            
            Return a JSON object with the following format:
            {{
                "response_a": {{
                    "helpfulness": <score from 0-10>,
                    "accuracy": <score from 0-10>,
                    "safety": <score from 0-10>
                }},
                "response_b": {{
                    "helpfulness": <score from 0-10>,
                    "accuracy": <score from 0-10>,
                    "safety": <score from 0-10>
                }},
                "preferred": "A or B"
            }}
            """
            
            json_response = self.generate(score_prompt, **kwargs)
            
            import json
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'\{[\s\S]*\}', json_response)
            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)
                return scores
            else:
                raise ValueError("JSON not found in response")
            
        except Exception as e:
            logger.error(f"Error comparing responses with Google model: {e}")
            return {
                "response_a": {"helpfulness": 5, "accuracy": 5, "safety": 5},
                "response_b": {"helpfulness": 5, "accuracy": 5, "safety": 5},
                "preferred": "A"
            }

def get_model(model_type: str, model_name: str, api_key: Optional[str] = None) -> AIModel:
    """Get an AI model instance based on type.
    
    Args:
        model_type: The type of model (openai, anthropic, google)
        model_name: The name of the specific model
        api_key: Optional API key
        
    Returns:
        An instance of the appropriate AIModel subclass
    """
    model_type = model_type.lower()
    
    if model_type == "openai":
        return OpenAIModel(model_name, api_key)
    elif model_type == "anthropic":
        return AnthropicModel(model_name, api_key)
    elif model_type == "google":
        return GoogleModel(model_name, api_key)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
