# Diffusion Model for Two Moons Distribution

This project implements a simple 2D diffusion model using PyTorch Lightning to learn and generate samples from the Two Moons distribution.

## Project Goals

- Train a diffusion model to gradually sample from a 2D Gaussian distribution to approximate the Two Moons distribution.
- Visualize and compare real and generated samples.

## Project Structure
.
├── diffusion_model.py # Main code file
├── requirements.txt # Dependencies list
├── README.md # Project description (this file)
├── two_moons_result.png # Example output plot saved after training

## Environment Requirements

- Python 3.8+
- PyTorch 1.13+
- PyTorch Lightning 2.0+
- scikit-learn
- matplotlib

Install dependencies with:

```bash
pip install -r requirements.txt

python ddmp_module.py


# Summary of Diffusion Model Project

## Project Overview
This project implements a diffusion model to generate samples approximating the Two Moons distribution using PyTorch Lightning.

## Methodology
- Model: MLP-based noise predictor taking 2D points and timestep embeddings.
- Noise schedule: Linear beta schedule with 100 timesteps.
- Training: 2000 epochs using Adam optimizer.
- Sampling: Euler method for reverse diffusion.

## Evaluation Metrics
We used the Maximum Mean Discrepancy (MMD) to quantify the similarity between generated and real samples. MMD measures the difference in distributions with lower values indicating better fit.

## Results
- Training loss converged steadily (plot attached).
- Final MMD score was  0.007543.
- The generated samples visually resemble the real Two Moons data (see `two_moons_result.png`).

## Improvements and Future Work
- Experiment with different noise schedules and model sizes.
- Explore other architectures like UNet for noise prediction.
- Try different sampling algorithms for better generation quality.

## Conclusion
The diffusion model successfully learned the complex Two Moons distribution, validating the approach for simple 2D generative modeling.



