# Diffusion Model Visualizer 🌊

Interactive visualization of diffusion
models — DDPM, DDIM and Stable Diffusion
architecture explained step by step.

## Live Demo
[Click here](YOUR_STREAMLIT_URL)

## Features
- Forward noising process visualization
- 3 noise schedules: linear, cosine, quadratic
- Reverse denoising animation
- UNet architecture diagram
- Score matching with vector field
- DDPM vs DDIM comparison (steps vs FID)
- Modern sampler comparison table
- 5 architecture breakdowns (SD, DALL-E 2 etc)
- Diffusion timeline 2020-2023
- CFG scale explanation
- Interview Q&A

## Tools Used
- Python, Streamlit, NumPy, SciPy,
  Plotly, Pandas

## How to Run Locally
pip install streamlit numpy plotly pandas scipy
streamlit run app.py
