from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# The submission here should simply be a function that returns a model
# compatible with scikit-learn API
# For regression task, we use RandomForestRegressor instead of Classifier
def get_model():
    """
    Returns a scikit-learn compatible regression model.
    
    This model includes:
    - Missing value imputation (median)
    - Feature scaling (standardization)
    - Random Forest Regressor with optimized hyperparameters
    """
    # Preprocessing pipeline
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Full pipeline with preprocessing and model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        ))
    ])
    
    return model
