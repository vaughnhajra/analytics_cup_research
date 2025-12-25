# Contributing Guidelines

Thank you for your interest in contributing! This project aims to move beyond static recognition tests toward true measures of on-field cognition using tracking data.

### Project Architecture
This project is divided into a core library (`/src`) and a research layer (`submission.ipynb`). To ensure modularity, please follow this structure:

- `src/data_loader.py`: Handles data ingestion and metadata extraction.
- `src/kinematics.py`: Contains the signal processing logic, including Savitzky-Golay filtering and vector differentiation.
- `src/feature_engineering.py`: The primary pipeline for transforming raw kinematics into model-ready features

### 1. Adding New Tactical Features
To add a new feature, do not add it directly to the notebook. Instead:

- Define the calculation function in `src/feature_engineering.py`.
- Register the feature within the `add_model_features` function.
- Ensure the feature is appended to the `cols_to_keep` list to make it available for the regression and Random Forest models.

### 2. Refining Response Identification
The current response dtection relies on a change in acelleration over a vector magnitude within a time window. 

- If you propose a more accurate threshold, update the defaults in `detect_player_responses` within `src/feature_engineering.py`.
- Provide statistical justification (e.g., improved R<sup>2</sup> or lower residuals) in your pull request.

### 3. Improving Signal Processing
Currently, a Savitzky-Golay filter is used for smoothing.

- New filters or extrapolation methods should be added to `compute_all_kinematics` in `src/kinematics.py`.
- Ensure that any new dependency is added to `requirements.txt`.

### Coding Standards
- Vectorization: Use NumPy for all kinematic calculations to maintain performance over large tracking datasets.

- Documentation: All new functions must include a docstring explaining the mathematical approach and unit of measurement.
