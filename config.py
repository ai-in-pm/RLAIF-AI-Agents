# Configuration for RLAIF implementation

# Model types
POLICY_MODEL_TYPE = "openai"  # Options: openai, anthropic, google
FEEDBACK_MODEL_TYPE = "openai"  # Options: openai, anthropic, google

# Model specific configurations
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
GOOGLE_MODEL = "gemini-1.5-pro"

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_EPOCHS = 10
VALIDATION_SPLIT = 0.1

# RLAIF parameters
CONSTITUTION_FILE = "constitution.txt"

# Paths
DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# Logging
LOG_LEVEL = "INFO"
