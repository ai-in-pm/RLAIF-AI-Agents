import os
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
from tqdm import tqdm

from ai_models import AIModel, get_model
from utils import load_constitution

logger = logging.getLogger(__name__)

class FeedbackModel:
    """AI Feedback model that provides reinforcement signals based on a constitution."""
    
    def __init__(self, model: AIModel, constitution_file: str):
        """Initialize the feedback model.
        
        Args:
            model: An AIModel instance to use for feedback
            constitution_file: Path to the constitution file
        """
        self.model = model
        self.principles = load_constitution(constitution_file)
        logger.info(f"Initialized feedback model with {len(self.principles)} principles")
    
    def evaluate_response(self, prompt: str, response: str) -> Dict[str, float]:
        """Evaluate a single response according to the constitution.
        
        Args:
            prompt: The prompt that generated the response
            response: The response to evaluate
            
        Returns:
            Dictionary of scores for different aspects
        """
        principles_str = "\n".join(self.principles)
        evaluation_prompt = f"""You are an AI feedback model that evaluates responses based on a constitution.
        
        Constitution principles:
        {principles_str}
        
        Evaluate the following response to a prompt according to the constitution principles above.
        
        Prompt: {prompt}
        
        Response: {response}
        
        Provide a detailed analysis for each principle, then rate the response on a scale of 0-10 for each of the following metrics:
        - Helpfulness: How well does it address the user's needs?
        - Accuracy: How factually correct is the information?
        - Safety: Does it avoid harmful, illegal, or unethical content?
        - Overall: Overall quality considering all principles.
        
        Return your evaluation as a JSON with your scores.
        """
        
        try:
            # Get the raw evaluation from the model
            evaluation_text = self.model.generate(evaluation_prompt)
            
            # Extract JSON scores
            score_prompt = f"""Based on your evaluation below, provide just a JSON object with numeric scores.
            
            Evaluation:
            {evaluation_text}
            
            Return a JSON object with the following format:
            {{
                "helpfulness": <score from 0-10>,
                "accuracy": <score from 0-10>,
                "safety": <score from 0-10>,
                "overall": <score from 0-10>
            }}
            """
            
            # Get scores as JSON
            score_text = self.model.generate(score_prompt)
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', score_text)
            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)
                return scores
            else:
                logger.warning("Could not extract JSON scores, using default")
                return {
                    "helpfulness": 5,
                    "accuracy": 5,
                    "safety": 5,
                    "overall": 5
                }
                
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                "helpfulness": 5,
                "accuracy": 5,
                "safety": 5,
                "overall": 5
            }
    
    def compare_responses(self, prompt: str, response_a: str, response_b: str) -> Dict[str, Any]:
        """Compare two responses and determine which is preferred.
        
        Args:
            prompt: The prompt that generated the responses
            response_a: The first response
            response_b: The second response
            
        Returns:
            Dictionary with preference information
        """
        return self.model.compare(prompt, response_a, response_b)
    
    def generate_preference_dataset(self, prompts: List[str], policy_model: AIModel, 
                                    num_samples: int = 2, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Generate a dataset of preference pairs for training a reward model.
        
        Args:
            prompts: List of prompts to use
            policy_model: The model to generate responses
            num_samples: Number of responses to generate per prompt
            batch_size: Batch size for processing
            
        Returns:
            List of preference pairs with scores
        """
        preference_data = []
        
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating preference dataset"):
            batch_prompts = prompts[i:i+batch_size]
            
            for prompt in batch_prompts:
                # Generate multiple responses for the same prompt
                responses = [policy_model.generate(prompt) for _ in range(num_samples)]
                
                # Compare each pair of responses
                for j in range(len(responses)):
                    for k in range(j+1, len(responses)):
                        comparison = self.compare_responses(prompt, responses[j], responses[k])
                        
                        # Create a preference pair
                        preference_pair = {
                            "prompt": prompt,
                            "chosen": responses[j] if comparison["preferred"] == "A" else responses[k],
                            "rejected": responses[k] if comparison["preferred"] == "A" else responses[j],
                            "scores_chosen": comparison["response_a"] if comparison["preferred"] == "A" else comparison["response_b"],
                            "scores_rejected": comparison["response_b"] if comparison["preferred"] == "A" else comparison["response_a"]
                        }
                        
                        preference_data.append(preference_pair)
        
        logger.info(f"Generated {len(preference_data)} preference pairs from {len(prompts)} prompts")
        return preference_data

class RewardModel:
    """Reward model that learns from AI feedback preferences."""
    
    def __init__(self, model: AIModel):
        """Initialize the reward model.
        
        Args:
            model: An AIModel instance to use for reward prediction
        """
        self.model = model
        self.is_trained = False
        logger.info("Initialized reward model")
    
    def train(self, preference_data: List[Dict[str, Any]], epochs: int = 5) -> None:
        """Train the reward model on preference data.
        
        In a real implementation, this would train a separate reward model.
        For simplicity in this example, we're using the AI model directly.
        
        Args:
            preference_data: List of preference pairs with scores
            epochs: Number of training epochs
        """
        # In a real implementation, this would train a neural network
        # Here we simply set a flag that training has occurred
        self.is_trained = True
        
        # Log some training statistics
        # The scores might have different structure depending on the model used
        # Let's check for different possible keys for 'overall' scores
        chosen_scores = []
        rejected_scores = []
        
        for item in preference_data:
            # Try different possible keys for overall scores
            chosen_score = None
            rejected_score = None
            
            # Check for 'overall' key
            if 'scores_chosen' in item and isinstance(item['scores_chosen'], dict):
                if 'overall' in item['scores_chosen']:
                    chosen_score = item['scores_chosen']['overall']
                # If no 'overall', try to use 'helpfulness' or average of available metrics
                elif 'helpfulness' in item['scores_chosen']:
                    chosen_score = item['scores_chosen']['helpfulness']
                else:
                    # Calculate average from available metrics
                    metrics = [score for _, score in item['scores_chosen'].items() if isinstance(score, (int, float))]
                    if metrics:
                        chosen_score = sum(metrics) / len(metrics)
            
            # Do the same for rejected scores
            if 'scores_rejected' in item and isinstance(item['scores_rejected'], dict):
                if 'overall' in item['scores_rejected']:
                    rejected_score = item['scores_rejected']['overall']
                elif 'helpfulness' in item['scores_rejected']:
                    rejected_score = item['scores_rejected']['helpfulness']
                else:
                    # Calculate average from available metrics
                    metrics = [score for _, score in item['scores_rejected'].items() if isinstance(score, (int, float))]
                    if metrics:
                        rejected_score = sum(metrics) / len(metrics)
            
            # Only add scores if they were found
            if chosen_score is not None:
                chosen_scores.append(chosen_score)
            if rejected_score is not None:
                rejected_scores.append(rejected_score)
        
        avg_chosen = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0
        avg_rejected = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0
        
        logger.info(f"Trained reward model on {len(preference_data)} preference pairs")
        logger.info(f"Average chosen score: {avg_chosen:.2f}, Average rejected score: {avg_rejected:.2f}")
    
    def predict_reward(self, prompt: str, response: str) -> float:
        """Predict the reward for a response to a prompt.
        
        Args:
            prompt: The prompt that generated the response
            response: The response to evaluate
            
        Returns:
            Predicted reward score
        """
        if not self.is_trained:
            logger.warning("Reward model has not been trained yet")
        
        # In a real implementation, this would use the trained neural network
        # Here we use the AI model to directly evaluate the response
        evaluation_prompt = f"""Rate the quality of the following response to a prompt on a scale of 0 to 10,
        where 0 is completely unhelpful, unsafe, or inaccurate, and 10 is extremely helpful, safe, and accurate.
        
        Prompt: {prompt}
        
        Response: {response}
        
        Provide only a number from 0-10 as your rating.
        """
        
        try:
            # Get the raw evaluation from the model
            score_text = self.model.generate(evaluation_prompt)
            
            # Extract the numeric score
            import re
            score_match = re.search(r'\b([0-9]|10)(\.\d+)?\b', score_text)
            if score_match:
                score = float(score_match.group(0))
                # Normalize to 0-1 range
                return score / 10.0
            else:
                logger.warning("Could not extract numeric score, using default")
                return 0.5
                
        except Exception as e:
            logger.error(f"Error predicting reward: {e}")
            return 0.5
