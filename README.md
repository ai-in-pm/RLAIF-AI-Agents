# RLAIF AI Agents

Reinforcement Learning from AI Feedback (RLAIF) implementation for training and fine-tuning AI agents.

## Overview

This project implements RLAIF, a technique that uses AI models to provide feedback and train other AI models. Instead of using human feedback (as in RLHF), this approach leverages AI-generated evaluations to create a scalable training pipeline.

### Key Features

- **Multiple AI Model Support**: Works with OpenAI, Anthropic, and Google's generative AI models
- **Constitution-Based Feedback**: Evaluates responses based on a set of principles defined in a constitution
- **Reward Modeling**: Implements preference-based learning through comparison of responses
- **Policy Optimization**: Trains models to generate better responses based on feedback
- **Training Data Generation**: Creates synthetic preference datasets for training

## Project Structure

- `main.py`: Main execution script that runs the RLAIF pipeline
- `config.py`: Configuration parameters for models, training, and file paths
- `constitution.txt`: Ethical principles guiding the AI feedback model
- `utils.py`: Utility functions for loading API keys and ensuring directory structure
- `ai_models.py`: Wrapper classes for different AI models (OpenAI, Anthropic, Google)
- `feedback_model.py`: Contains the `FeedbackModel` and `RewardModel` classes
- `policy_model.py`: Defines the `PolicyModel` class for generating and optimizing responses

## Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for at least one of the supported AI services (OpenAI, Anthropic, Google)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd RLAIF-AI-Agents
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
   Or install them individually:
   ```
   pip install python-dotenv openai anthropic google-generativeai numpy matplotlib tqdm torch torchvision
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

## Usage

### Running the RLAIF Pipeline

```
python main.py
```

The script will:
1. Initialize AI models for generating responses and providing feedback
2. Create a set of sample prompts for training
3. Generate preference datasets by comparing response quality
4. Train a reward model based on these preferences
5. Optimize a policy model using the reward model
6. Demonstrate the improved responses

### Command Line Arguments

- `--direct`: Use the direct RLAIF approach instead of the traditional one with a separate reward model
- `--model_type TYPE`: Choose between 'openai', 'anthropic', or 'google' (default: 'openai')
- `--feedback_model_type TYPE`: Specify a different model for feedback (default: same as model_type)

## How It Works

### RLAIF Process

1. **Initial Response Generation**: The policy model generates responses to prompts
2. **Feedback Generation**: The feedback model evaluates responses based on principles
3. **Preference Creation**: Multiple responses are compared to create preference pairs
4. **Reward Model Training**: A model learns to predict which responses are preferred
5. **Policy Optimization**: The policy model is fine-tuned to maximize the reward

### Advantages over RLHF

- **Scalability**: Can generate and evaluate more data without human bottlenecks
- **Cost-effectiveness**: Reduces the need for human evaluators
- **Consistency**: AI feedback is more consistent across evaluations
- **Speed**: Faster training cycles compared to human-in-the-loop approaches

## Sample Outputs

After running the RLAIF pipeline, you can find the following output files:

- `data/preference_data.json`: Contains the generated preference pairs used for training
- `outputs/demo_responses.json`: Contains example prompts and optimized responses

## Troubleshooting

### Common Issues

- **ModuleNotFoundError**: Ensure all dependencies are installed in your virtual environment by running `pip install -r requirements.txt`
- **API Key Errors**: Verify that your `.env` file contains valid API keys and is located in the project root
- **KeyError in score processing**: If you encounter errors related to score key names, check the feedback model's evaluation format

### Debugging

You can enable more detailed logging by setting the logging level in `utils.py`:

```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
