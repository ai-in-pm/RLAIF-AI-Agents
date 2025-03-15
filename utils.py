import os
import logging
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_api_keys() -> Dict[str, str]:
    """Load API keys from environment variables."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API keys
    api_keys = {
        'openai': os.getenv('OPENAI_API_KEY'),
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    }
    
    # Verify all keys are loaded
    for service, key in api_keys.items():
        if not key:
            logger.warning(f"API key for {service} not found in environment variables")
        else:
            logger.info(f"API key for {service} loaded successfully")
    
    return api_keys

def load_constitution(file_path: str) -> List[str]:
    """Load constitution principles from a file."""
    try:
        with open(file_path, 'r') as f:
            # Skip comments and empty lines
            lines = [line.strip() for line in f.readlines() 
                    if line.strip() and not line.strip().startswith('#')]
        
        # Extract numbered principles
        principles = []
        for line in lines:
            if any(line.startswith(f"{i}.") for i in range(1, 100)):
                principles.append(line)
        
        logger.info(f"Loaded {len(principles)} principles from constitution")
        return principles
    except Exception as e:
        logger.error(f"Error loading constitution: {e}")
        return []

def ensure_directories(dirs: List[str]) -> None:
    """Ensure all required directories exist."""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory ensured: {dir_path}")
