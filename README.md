# Piecewise Linear Recurrent Neural Networks (PLRNN) for State-Space Modeling

This repository contains the implementation of Piecewise Linear Recurrent Neural Networks (PLRNNs) developed for complex time-series analysis and state-space modeling. The project was built using MATLAB and focuses on extracting underlying dynamic states from high-dimensional data, specifically incorporating biophysical constraints.

## Project Overview

Recurrent Neural Networks are highly effective for modeling sequential data, but interpreting their internal dynamics can be challenging. This project utilizes the PLRNN architecture, which provides a mathematically tractable framework for dynamical systems analysis while maintaining high representational capacity. 

The codebase explores both standard PLRNN State-Space Models (SSM) and regularized versions applied to specific data types, such as BOLD (Blood-Oxygen-Level-Dependent) signals.

## Repository Structure

* /code_PLRNN_SSM - Core implementation of the standard Piecewise Linear RNN state-space models.
* /code_PLRNNreg_BOLD_SSM - Regularized PLRNN models tailored for analyzing BOLD signal dynamics.
* /Published_code - Finalized, clean scripts prepared for deployment and review.
* /data - Contains sample `.mat` datasets used for training and evaluating the models.
* /resources - Additional documentation, biophysical constraints configurations, and project metadata.

## Technical Stack
* Language: MATLAB
* Concepts: Machine Learning, Recurrent Neural Networks (RNN), Dynamical Systems, State-Space Modeling (SSM).

## Note on Repository Content
To maintain a clean and accessible repository, OS-specific hidden files (e.g., macOS `.DS_Store`), MATLAB auto-generated cache folders, and heavy data matrices have been excluded via `.gitignore`.
