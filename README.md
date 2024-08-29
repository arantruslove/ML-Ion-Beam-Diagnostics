# Machine Learning Guided Diagnostics for Laser-Driven Ion Beams

## Overview
This project explores the application of machine learning, specifically convolutional neural networks (CNNs), to rapidly diagnose key parameters of laser-driven ion beams. By using synthetic data generation and Bayesian optimization, the project aims to predict three essential beam parameters: maximum proton energy (Emax), proton spectra temperature (Tp), and the number of protons in the beam (N0). The developed methods are intended to improve the efficiency and accuracy of beam diagnostics in advanced laser-driven ion acceleration systems.

## Project Objectives
- **Synthetic Data Generation:** Develop a Monte Carlo simulation to generate synthetic images of proton beams passing through a 3x3 grid of aluminum filters and a scintillator.
- **Neural Network Training:** Train a CNN on the synthetic dataset to predict beam parameters from CCD-captured images.
- **Bayesian Optimization:** Optimize both the filter thicknesses used in the data generation process and the architecture of the CNN to enhance prediction accuracy.

## Methodology
1. **Data Generation:** 
   - Simulated 100,000 synthetic images using a model that accounts for proton energy distributions and energy deposition in filters and scintillators.
   - Applied Gaussian noise and normalization techniques to mimic real-world data collection.

2. **Neural Network:**
   - Implemented a CNN using Keras, optimized through mean-squared error loss minimization.
   - Trained on 75,000 images and validated on 25,000 images.

3. **Bayesian Optimization:**
   - Used Optuna to optimize filter thickness configurations and CNN architectural parameters, improving model accuracy significantly.

## Results
- **Performance Improvements:** Bayesian optimization led to a 40% reduction in prediction error when optimizing filter thicknesses and an additional 10% improvement when optimizing the CNN architecture.
- **Prediction Accuracy:** The optimized CNN achieved mean relative absolute errors (MRAE) of 7% for Emax, 13% for Tp, and 4% for N0.
- **Speed:** The optimized CNN's prediction time averaged 7.3 ms, making it suitable for real-time applications in high-frequency laser systems.

## Conclusion
The project demonstrates the potential of machine learning to significantly improve the speed and accuracy of diagnostics in laser-driven ion beam systems. The optimized CNN model shows promising results for real-time applications, paving the way for further research and development.
