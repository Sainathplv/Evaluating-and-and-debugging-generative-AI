# Evaluating-and-and-debugging-generative-AI
This repository contains a series of Jupyter notebooks that demonstrate the training of diffusion models, sampling from diffusion models, and fine-tuning a language model for character backstory generation. We use Weights & Biases (W&B) to track and visualize important metrics, compare experiments, and collaborate effectively.

# 01_intro_starter.ipynb
Introduces the use of wandb (Weights & Biases) for tracking and visualizing important metrics during the training of a sprite classification model. It demonstrates how to set up W&B for monitoring model performance, comparing experiments, collaborating with team members, and reproducing results.
A simple multi-layer perceptron (MLP) model is defined for classifying sprites. The model consists of two linear layers with ReLU activation and dropout for regularization.
It uses a SimpleNamespace object to store hyperparameters such as the number of epochs, batch size, learning rate, and dropout rate.
The training loop involves iterating over epochs, processing batches of data, calculating loss, and updating model parameters. Metrics such as loss and accuracy are logged to W&B for visualization and analysis.

# 02_diffusion_training_starter.ipynb
It focuses on training a diffusion model with the integration of Weights & Biases (W&B) for logging and tracking purposes. It is based on the Lab3 notebook from the "How diffusion models work" course. The notebook demonstrates how to log training loss and metrics, sample from the model during training, and upload samples and model checkpoints to W&B.

# 03_diffusion_sampling_starter.ipynb
It focuses on sampling from a previously trained diffusion model. It compares samples from two different samplers: Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM). Additionally, it demonstrates how to visualize mixing samples with conditional diffusion models.
It explains  how to sample images from the diffusion model using both DDPM and DDIM samplers. It also demonstrates how to mix samples using conditional diffusion models.

# 04_llm_eval_starter.ipynb
It focuses on evaluating and tracing Large Language Models (LLMs) using Weights & Biases (W&B). It demonstrates how to use W&B Tables for evaluating the generations produced by OpenAI's GPT-3.5 model.It also demonstrates how to make calls to the OpenAI LLM to generate names for game assets. It uses the retry decorator from tenacity to handle API call retries with exponential backoff and how to use W&B Tables to log and evaluate the generated names. 

# 05_train_llm_starter.ipynb
It demonstrates how to fine-tune a language model to generate character backstories using the HuggingFace Trainer with W&B integration. It uses a tiny language model (TinyStories-33M) due to resource constraints, but the techniques shown are applicable to larger models as well.
It sets up the TrainingArguments and uses the Trainer class from HuggingFace to fine-tune the language model on the prepared dataset. Key training arguments and hyperparameters are configured for effective fine-tuning.




