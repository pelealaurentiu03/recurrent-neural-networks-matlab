# Piecewise Linear Recurrent Neural Networks (PLRNN) for State-Space Modeling

This repository contains the implementation of Piecewise Linear Recurrent Neural Networks (PLRNNs) developed for complex time-series analysis and state-space modeling. The project was built using MATLAB and focuses on extracting underlying dynamic states from high-dimensional data, specifically incorporating biophysical constraints.

## Project Overview

Recurrent Neural Networks are highly effective for modeling sequential data, but interpreting their internal dynamics can be challenging. This project utilizes the PLRNN architecture, which provides a mathematically tractable framework for dynamical systems analysis while maintaining high representational capacity. 

The codebase explores both standard PLRNN State-Space Models (SSM) and regularized versions applied to functional neuroimaging data. Specifically, the models are tailored for analyzing BOLD (Blood-Oxygen-Level-Dependent) signal dynamics, utilizing the standard **Schaefer 2018 brain parcellation atlas** to extract and process regional time-series data.

## Repository Structure

* /code_PLRNN_SSM - Core implementation of the standard Piecewise Linear RNN state-space models.
* /code_PLRNNreg_BOLD_SSM - Regularized PLRNN models tailored for analyzing BOLD signal dynamics.
* /figures - Finalized, clean representations of what the code calculated.

## Technical Stack
* Language: MATLAB
* Concepts: Machine Learning, Recurrent Neural Networks (RNN), Dynamical Systems, State-Space Modeling (SSM).

## Note on Repository Content
To maintain a clean and accessible repository, OS-specific hidden files (e.g., macOS `.DS_Store`), MATLAB auto-generated cache folders, and heavy data matrices have been excluded via `.gitignore`.
