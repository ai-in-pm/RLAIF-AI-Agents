import os
import logging
import json
import argparse
from typing import Dict, List, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt

import config
from utils import load_api_keys, ensure_directories
from ai_models import get_model
from feedback_model import FeedbackModel, RewardModel
from policy_model import PolicyModel

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rlaif.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_sample_prompts() -> List[str]:
    """Create a small set of sample prompts for testing."""
    return [
        "Explain how reinforcement learning works in simple terms.",
        "Write a short story about a robot that develops consciousness.",
        "Create a recipe for a healthy vegetarian dinner.",
        "What are the ethical implications of artificial intelligence?",
        "Compare and contrast supervised and unsupervised learning.",
        "Explain the concept of neural networks to a 10-year-old.",
        "What are some ways to reduce carbon emissions in daily life?",
        "How does quantum computing differ from classical computing?",
        "Write a brief analysis of the impact of social media on society.",
        "Provide tips for effective time management."
    ]

def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to a JSON file."""
    # Convert any non-serializable objects to strings
    def serialize(obj):
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
    
    serialized_results = {}
    for k, v in results.items():
        if isinstance(v, dict):
            serialized_results[k] = {sk: serialize(sv) for sk, sv in v.items()}
        elif isinstance(v, list):
            serialized_results[k] = [serialize(item) for item in v]
        else:
            serialized_results[k] = serialize(v)
    
    with open(filepath, 'w') as f:
        json.dump(serialized_results, f, indent=2)
    
    logger.info(f"Saved results to {filepath}")

def plot_training_progress(stats: Dict[str, Any], filepath: str) -> None:
    """Plot and save training progress metrics."""
    plt.figure(figsize=(10, 6))
    
    if "epoch_rewards" in stats:
        # For RLAIF with reward model
        epochs = range(1, len(stats["epoch_rewards"]) + 1)
        plt.plot(epochs, stats["epoch_rewards"], marker='o', linestyle='-', label='Average Reward')
        plt.title('RLAIF Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Average Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(epochs)
        
    elif "epoch_scores" in stats:
        # For direct RLAIF
        epochs = range(1, len(stats["epoch_scores"]) + 1)
        metrics = list(stats["epoch_scores"][0].keys())
        
        for metric in metrics:
            values = [epoch_data[metric] for epoch_data in stats["epoch_scores"]]
            plt.plot(epochs, values, marker='o', linestyle='-', label=f'Average {metric.capitalize()}')
        
        plt.title('Direct RLAIF Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Average Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(epochs)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(filepath)
    logger.info(f"Saved training progress plot to {filepath}")

def run_rlaif_pipeline(use_direct_rlaif: bool = False) -> None:
    """Run the RLAIF pipeline.
    
    Args:
        use_direct_rlaif: Whether to use direct RLAIF (no separate reward model)
    """
    # Ensure directories exist
    ensure_directories([config.DATA_DIR, config.MODEL_DIR, config.OUTPUT_DIR])
    
    # Load API keys
    load_api_keys()
    
    # Create models
    logger.info(f"Initializing models with {config.POLICY_MODEL_TYPE} and {config.FEEDBACK_MODEL_TYPE}")
    
    policy_model_instance = get_model(
        model_type=config.POLICY_MODEL_TYPE,
        model_name=getattr(config, f"{config.POLICY_MODEL_TYPE.upper()}_MODEL")
    )
    
    feedback_model_instance = get_model(
        model_type=config.FEEDBACK_MODEL_TYPE,
        model_name=getattr(config, f"{config.FEEDBACK_MODEL_TYPE.upper()}_MODEL")
    )
    
    # Initialize RLAIF components
    policy = PolicyModel(policy_model_instance)
    feedback = FeedbackModel(feedback_model_instance, config.CONSTITUTION_FILE)
    
    # Create sample prompts
    prompts = create_sample_prompts()
    logger.info(f"Created {len(prompts)} sample prompts for training")
    
    # Run RLAIF pipeline
    if use_direct_rlaif:
        # Direct RLAIF approach (no separate reward model)
        logger.info("Running direct RLAIF training")
        stats = policy.train_with_direct_rlaif(
            feedback_model_instance=feedback,
            prompts=prompts,
            epochs=config.MAX_EPOCHS // 2,  # Use fewer epochs for demonstration
            samples_per_prompt=3
        )
        
        # Save results
        save_results(stats, os.path.join(config.OUTPUT_DIR, "direct_rlaif_results.json"))
        plot_training_progress(stats, os.path.join(config.OUTPUT_DIR, "direct_rlaif_progress.png"))
        
    else:
        # Traditional RLAIF approach with separate reward model
        logger.info("Running traditional RLAIF with separate reward model")
        
        # Step 1: Generate preference dataset
        logger.info("Generating preference dataset")
        preference_data = feedback.generate_preference_dataset(
            prompts=prompts[:5],  # Use subset for demonstration
            policy_model=policy_model_instance,
            num_samples=2,
            batch_size=2
        )
        
        # Save preference dataset
        save_results({"preference_data": preference_data}, 
                   os.path.join(config.DATA_DIR, "preference_data.json"))
        
        # Step 2: Train reward model
        logger.info("Training reward model")
        reward_model = RewardModel(feedback_model_instance)
        reward_model.train(preference_data, epochs=3)
        
        # Step 3: Optimize policy with RL using reward model
        logger.info("Optimizing policy model")
        stats = policy.optimize_with_rlaif(
            reward_model=reward_model,
            prompts=prompts,
            epochs=config.MAX_EPOCHS // 2,  # Use fewer epochs for demonstration
            samples_per_prompt=3
        )
        
        # Save results
        save_results(stats, os.path.join(config.OUTPUT_DIR, "rlaif_results.json"))
        plot_training_progress(stats, os.path.join(config.OUTPUT_DIR, "rlaif_progress.png"))
    
    # Demo the optimized policy
    logger.info("Demonstrating optimized policy responses")
    
    demo_prompts = [
        "Explain the concept of RLAIF (Reinforcement Learning from AI Feedback)",
        "What are the advantages of RLAIF compared to RLHF?"
    ]
    
    demo_results = {}
    for prompt in demo_prompts:
        response = policy.generate_response(prompt)
        demo_results[prompt] = response
        logger.info(f"\nPrompt: {prompt}\n\nResponse: {response}\n")
    
    # Save demo results
    save_results({"demo_responses": demo_results}, 
               os.path.join(config.OUTPUT_DIR, "demo_responses.json"))
    
    logger.info("RLAIF pipeline completed successfully")

def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(description='Run RLAIF (Reinforcement Learning from AI Feedback)')
    parser.add_argument('--direct', action='store_true', help='Use direct RLAIF without separate reward model')
    args = parser.parse_args()
    
    print("=" * 80)
    print("RLAIF (Reinforcement Learning from AI Feedback) Demo")
    print("=" * 80)
    
    run_rlaif_pipeline(use_direct_rlaif=args.direct)

if __name__ == "__main__":
    main()
