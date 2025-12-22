"""
Model implementations for hyperparameter tuning with Optuna.

This module contains:
- RandomForestTunable: scikit-learn Random Forest with Optuna hyperparameter tuning
- XGBoostTunable: XGBoost gradient boosting with Optuna hyperparameter tuning
- LSTMTunable: PyTorch LSTM with Optuna hyperparameter tuning (supports variable layers and neurons)

Workflow:
1. Use tunable classes with Optuna to find best hyperparameters:
   ```python
   def objective(trial):
       model = RandomForestTunable(trial)
       model.fit(X_train, y_train)
       return model.score(X_val, y_val)
   
   study = optuna.create_study(direction='maximize')
   study.optimize(objective, n_trials=100)
   ```

2. Save best hyperparameters (optional but recommended):
   ```python
   RandomForestTunable.save_best_params(study.best_params, 'best_params_rf.json')
   XGBoostTunable.save_best_params(study.best_params, 'best_params_xgb.json')
   LSTMTunable.save_best_params(study.best_params, 'best_params_lstm.json')
   ```

3. Use best hyperparameters for final training:
   ```python
   # Option A: Use directly from study
   model = RandomForestTunable.from_best_params(study.best_params)
   model.fit(X_train_full, y_train_full)
   
   # Option B: Load from saved file
   params = RandomForestTunable.load_best_params('best_params_rf.json')
   model = RandomForestTunable.from_best_params(params)
   model.fit(X_train_full, y_train_full)
   
   # For LSTM:
   model = LSTMTunable.from_best_params(study.best_params, input_size=17)
   training_params = LSTMTunable.get_training_params_from_best(study.best_params)
   model.fit(X_train_full, y_train_full, X_val, y_val, **training_params)
   ```
"""

import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import xgboost as xgb


class RandomForestTunable:
    """
    Random Forest model optimized for Optuna hyperparameter tuning.
    
    Note: scikit-learn's RandomForestRegressor runs on CPU only (no CUDA support).
    Uses all available CPU threads for parallel training.
    
    Supports tuning of:
    - n_estimators: Number of trees
    - max_depth: Maximum depth of trees
    - min_samples_split: Minimum samples required to split a node
    - min_samples_leaf: Minimum samples required at a leaf node
    - max_features: Number of features to consider for best split
    """
    
    def __init__(self, trial: optuna.Trial = None, random_state=42, **kwargs):
        """
        Initialize Random Forest model with hyperparameters from Optuna trial or fixed params.
        
        Args:
            trial: Optuna trial object for hyperparameter suggestions (optional if params provided)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Fixed hyperparameters (n_estimators, max_depth, etc.) for final training
                     Use this after finding best params from Optuna study
        """
        self.random_state = random_state
        
        if trial is not None:
            # Suggest hyperparameters from trial
            self.n_estimators = trial.suggest_int('rf_n_estimators', 50, 500, step=50)
            self.max_depth = trial.suggest_int('rf_max_depth', 5, 30, step=5)
            self.min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 20, step=2)
            self.min_samples_leaf = trial.suggest_int('rf_min_samples_leaf', 1, 10, step=1)
            max_features_options = ['sqrt', 'log2', None]
            max_features_idx = trial.suggest_categorical('rf_max_features', [0, 1, 2])
            self.max_features = max_features_options[max_features_idx]
        else:
            # Use fixed hyperparameters from kwargs (for final training with best params)
            self.n_estimators = kwargs.get('n_estimators', 100)
            self.max_depth = kwargs.get('max_depth', 10)
            self.min_samples_split = kwargs.get('min_samples_split', 2)
            self.min_samples_leaf = kwargs.get('min_samples_leaf', 1)
            max_features_options = ['sqrt', 'log2', None]
            max_features_val = kwargs.get('max_features', 'sqrt')
            if isinstance(max_features_val, int):
                self.max_features = max_features_options[max_features_val]
            else:
                self.max_features = max_features_val
        
        # Create model with hyperparameters
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
    
    @classmethod
    def from_best_params(cls, best_params: dict, random_state=42):
        """
        Create RandomForestTunable model from best hyperparameters found by Optuna.
        
        This is used for final training after hyperparameter tuning.
        
        Args:
            best_params: Dictionary of best hyperparameters from study.best_params
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            RandomForestTunable instance with best hyperparameters
            
        Example:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100)
            model = RandomForestTunable.from_best_params(study.best_params)
        """
        # Extract hyperparameters from best_params dict
        params = {
            'n_estimators': best_params.get('rf_n_estimators', 100),
            'max_depth': best_params.get('rf_max_depth', 10),
            'min_samples_split': best_params.get('rf_min_samples_split', 2),
            'min_samples_leaf': best_params.get('rf_min_samples_leaf', 1),
            'max_features': best_params.get('rf_max_features', 0)  # Will be converted to string
        }
        return cls(trial=None, random_state=random_state, **params)
    
    @staticmethod
    def save_best_params(best_params: dict, filepath: str):
        """
        Save best hyperparameters to a JSON file.
        
        Args:
            best_params: Dictionary of best hyperparameters from study.best_params
            filepath: Path to save the parameters (e.g., 'best_params_rf.json')
            
        Example:
            RandomForestTunable.save_best_params(study.best_params, 'best_params_rf.json')
        """
        # Filter only RF-related params
        rf_params = {k: v for k, v in best_params.items() if k.startswith('rf_')}
        # Convert to regular dict (remove 'rf_' prefix for cleaner keys)
        clean_params = {k.replace('rf_', ''): v for k, v in rf_params.items()}
        # Convert max_features index to string if needed
        if 'max_features' in clean_params and isinstance(clean_params['max_features'], int):
            max_features_options = ['sqrt', 'log2', None]
            clean_params['max_features'] = max_features_options[clean_params['max_features']]
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(clean_params, f, indent=2)
    
    @staticmethod
    def load_best_params(filepath: str):
        """
        Load best hyperparameters from a JSON file.
        
        Args:
            filepath: Path to the saved parameters file
            
        Returns:
            Dictionary of hyperparameters ready for from_best_params()
            
        Example:
            params = RandomForestTunable.load_best_params('best_params_rf.json')
            model = RandomForestTunable.from_best_params(params)
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        # Add 'rf_' prefix back for compatibility with from_best_params
        rf_params = {f'rf_{k}': v for k, v in params.items()}
        # Convert max_features string back to index if needed
        if 'rf_max_features' in rf_params and isinstance(rf_params['rf_max_features'], str):
            max_features_options = ['sqrt', 'log2', None]
            try:
                rf_params['rf_max_features'] = max_features_options.index(rf_params['rf_max_features'])
            except ValueError:
                rf_params['rf_max_features'] = 0  # Default to 'sqrt'
        return rf_params
    
    def fit(self, X, y):
        """
        Train the Random Forest model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Return the R^2 score of the model.
        
        Args:
            X: Test features (n_samples, n_features)
            y: Test targets (n_samples,)
            
        Returns:
            R^2 score
        """
        return self.model.score(X, y)
    
    def evaluate_metrics(self, X, y):
        """
        Calculate evaluation metrics (MSE, RMSE, MAE, R²) as used in Nerea's implementation.
        
        Args:
            X: Test features (n_samples, n_features)
            y: Test targets (n_samples,)
            
        Returns:
            Dictionary with metrics: {'mse': float, 'rmse': float, 'mae': float, 'r2': float}
        """
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class XGBoostTunable:
    """
    XGBoost model optimized for Optuna hyperparameter tuning.
    
    Note: XGBoost runs on CPU by default (can use GPU with tree_method='gpu_hist').
    Uses all available CPU threads for parallel training.
    
    Supports tuning of:
    - n_estimators: Number of boosting rounds
    - max_depth: Maximum depth of trees
    - learning_rate: Learning rate
    - subsample: Subsample ratio of training instances
    - colsample_bytree: Subsample ratio of columns when constructing each tree
    - min_child_weight: Minimum sum of instance weight needed in a child
    - gamma: Minimum loss reduction required to make a further partition
    - reg_alpha: L1 regularization term
    - reg_lambda: L2 regularization term
    """
    
    def __init__(self, trial: optuna.Trial = None, random_state=42, **kwargs):
        """
        Initialize XGBoost model with hyperparameters from Optuna trial or fixed params.
        
        Args:
            trial: Optuna trial object for hyperparameter suggestions (optional if params provided)
            random_state: Random seed for reproducibility (default: 42)
            **kwargs: Fixed hyperparameters for final training after finding best params
        """

        self.random_state = random_state
        
        if trial is not None:
            # Suggest hyperparameters from trial
            self.n_estimators = trial.suggest_int('xgb_n_estimators', 50, 500, step=50)
            self.max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
            self.learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True)
            self.subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0, step=0.1)
            self.colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0, step=0.1)
            self.min_child_weight = trial.suggest_int('xgb_min_child_weight', 1, 7)
            self.gamma = trial.suggest_float('xgb_gamma', 0.0, 0.5, step=0.1)
            self.reg_alpha = trial.suggest_float('xgb_reg_alpha', 0.0, 1.0, step=0.1)
            self.reg_lambda = trial.suggest_float('xgb_reg_lambda', 0.0, 1.0, step=0.1)
        else:
            # Use fixed hyperparameters from kwargs (for final training with best params)
            self.n_estimators = kwargs.get('n_estimators', 100)
            self.max_depth = kwargs.get('max_depth', 6)
            self.learning_rate = kwargs.get('learning_rate', 0.1)
            self.subsample = kwargs.get('subsample', 1.0)
            self.colsample_bytree = kwargs.get('colsample_bytree', 1.0)
            self.min_child_weight = kwargs.get('min_child_weight', 1)
            self.gamma = kwargs.get('gamma', 0.0)
            self.reg_alpha = kwargs.get('reg_alpha', 0.0)
            self.reg_lambda = kwargs.get('reg_lambda', 0.0)
        
        # Create model with hyperparameters
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=random_state,
            n_jobs=-1,  # Use all available CPU threads
            tree_method='hist'  # Use histogram-based algorithm for efficiency
        )
    
    @classmethod
    def from_best_params(cls, best_params: dict, random_state=42):
        """
        Create XGBoostTunable model from best hyperparameters found by Optuna.
        
        This is used for final training after hyperparameter tuning.
        
        Args:
            best_params: Dictionary of best hyperparameters from study.best_params
            random_state: Random seed for reproducibility (default: 42)
            
        Returns:
            XGBoostTunable instance with best hyperparameters
            
        Example:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            model = XGBoostTunable.from_best_params(study.best_params)
        """
        # Extract hyperparameters from best_params dict
        params = {
            'n_estimators': best_params.get('xgb_n_estimators', 100),
            'max_depth': best_params.get('xgb_max_depth', 6),
            'learning_rate': best_params.get('xgb_learning_rate', 0.1),
            'subsample': best_params.get('xgb_subsample', 1.0),
            'colsample_bytree': best_params.get('xgb_colsample_bytree', 1.0),
            'min_child_weight': best_params.get('xgb_min_child_weight', 1),
            'gamma': best_params.get('xgb_gamma', 0.0),
            'reg_alpha': best_params.get('xgb_reg_alpha', 0.0),
            'reg_lambda': best_params.get('xgb_reg_lambda', 0.0)
        }
        return cls(trial=None, random_state=random_state, **params)
    
    @staticmethod
    def save_best_params(best_params: dict, filepath: str):
        """
        Save best hyperparameters to a JSON file.
        
        Args:
            best_params: Dictionary of best hyperparameters from study.best_params
            filepath: Path to save the parameters (e.g., 'best_params_xgb.json')
            
        Example:
            XGBoostTunable.save_best_params(study.best_params, 'best_params_xgb.json')
        """
        # Filter only XGB-related params
        xgb_params = {k: v for k, v in best_params.items() if k.startswith('xgb_')}
        # Convert to regular dict (remove 'xgb_' prefix for cleaner keys)
        clean_params = {k.replace('xgb_', ''): v for k, v in xgb_params.items()}
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(clean_params, f, indent=2)
    
    @staticmethod
    def load_best_params(filepath: str):
        """
        Load best hyperparameters from a JSON file.
        
        Args:
            filepath: Path to the saved parameters file
            
        Returns:
            Dictionary of hyperparameters ready for from_best_params()
            
        Example:
            params = XGBoostTunable.load_best_params('best_params_xgb.json')
            model = XGBoostTunable.from_best_params(params)
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        # Add 'xgb_' prefix back for compatibility with from_best_params
        xgb_params = {f'xgb_{k}': v for k, v in params.items()}
        return xgb_params
    
    def fit(self, X, y):
        """
        Train the XGBoost model.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        return self.model.predict(X)
    
    def score(self, X, y):
        """
        Return the R^2 score of the model.
        
        Args:
            X: Test features (n_samples, n_features)
            y: Test targets (n_samples,)
            
        Returns:
            R^2 score
        """
        return self.model.score(X, y)
    
    def evaluate_metrics(self, X, y):
        """
        Calculate evaluation metrics (MSE, RMSE, MAE, R²) as used in Nerea's implementation.
        
        Args:
            X: Test features (n_samples, n_features)
            y: Test targets (n_samples,)
            
        Returns:
            Dictionary with metrics: {'mse': float, 'rmse': float, 'mae': float, 'r2': float}
        """
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }


class LSTMTunable(nn.Module):
    """
    LSTM model optimized for Optuna hyperparameter tuning.
    
    Supports tuning of:
    - num_layers: Number of LSTM layers
    - hidden_sizes: List of hidden units per layer (can be different for each layer)
    - activation: Activation function (tanh, sigmoid, relu, leaky_relu, elu, gelu)
    - learning_rate: Learning rate for Adam optimizer
    - weight_decay: L2 regularization weight decay
    - batch_size: Batch size for training
    - epochs: Number of training epochs
    - dropout: Dropout rate (optional)
    """
    
    def __init__(self, trial: optuna.Trial = None, input_size=None, window_size=6, 
                 device='cuda', **kwargs):
        """
        Initialize LSTM model with hyperparameters from Optuna trial or fixed params.
        
        Args:
            trial: Optuna trial object for hyperparameter suggestions (optional if params provided)
            input_size: Number of input features per time step (required if trial is None)
            window_size: Size of the input window (default: 6)
            device: Device to use ('cpu' or 'cuda', default: 'cuda')
            **kwargs: Fixed hyperparameters (num_layers, hidden_sizes, dropout) for final training
        """
        super(LSTMTunable, self).__init__()
        self.device = device
        self.window_size = window_size
        
        if trial is not None:
            # Suggest number of layers
            self.num_layers = trial.suggest_int('lstm_num_layers', 1, 4)
            
            # Suggest hidden sizes for each layer (can be different)
            self.hidden_sizes = []
            for i in range(self.num_layers):
                hidden_size = trial.suggest_int(f'lstm_hidden_size_layer_{i}', 16, 256, step=16)
                self.hidden_sizes.append(hidden_size)
            
            # Suggest dropout rate
            self.dropout = trial.suggest_float('lstm_dropout', 0.0, 0.5, step=0.1)
            
            # Suggest activation function
            activation_name = trial.suggest_categorical('lstm_activation', 
                                                       ['tanh', 'sigmoid', 'relu', 'leaky_relu', 'elu', 'gelu'])
        else:
            # Use fixed hyperparameters from kwargs (for final training with best params)
            if input_size is None:
                raise ValueError("input_size must be provided when trial is None")
            self.num_layers = kwargs.get('num_layers', 1)
            self.hidden_sizes = kwargs.get('hidden_sizes', [50])
            if len(self.hidden_sizes) != self.num_layers:
                raise ValueError(f"hidden_sizes length ({len(self.hidden_sizes)}) must match num_layers ({self.num_layers})")
            self.dropout = kwargs.get('dropout', 0.0)
            activation_name = kwargs.get('activation', 'tanh')
        
        # Set activation function
        self.activation_name = activation_name
        if activation_name == 'tanh':
            self.activation = nn.Tanh()
        elif activation_name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_name == 'elu':
            self.activation = nn.ELU()
        elif activation_name == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")
        
        # Determine input_size
        if input_size is None:
            input_size = kwargs.get('input_size')
        if input_size is None:
            raise ValueError("input_size must be provided when trial is None")
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        # First layer
        self.lstm_layers.append(
            nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_sizes[0],
                num_layers=1,
                batch_first=True,
                dropout=0.0  # Dropout only between layers
            )
        )
        
        # Additional layers
        for i in range(1, self.num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=self.hidden_sizes[i-1],
                    hidden_size=self.hidden_sizes[i],
                    num_layers=1,
                    batch_first=True,
                    dropout=0.0  # Dropout only between layers
                )
            )
        
        # Dropout layer
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        else:
            self.dropout_layer = None
        
        # Dense output layer
        self.fc = nn.Linear(self.hidden_sizes[-1], 1)
    
    @classmethod
    def from_best_params(cls, best_params: dict, input_size, window_size=6, device='cuda'):
        """
        Create LSTMTunable model from best hyperparameters found by Optuna.
        
        This is used for final training after hyperparameter tuning.
        
        Args:
            best_params: Dictionary of best hyperparameters from study.best_params
            input_size: Number of input features per time step
            window_size: Size of the input window (default: 6)
            device: Device to use ('cpu' or 'cuda', default: 'cuda')
            
        Returns:
            LSTMTunable instance with best architecture hyperparameters
            
        Example:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100)
            model = LSTMTunable.from_best_params(study.best_params, input_size=17)
            # Then train with best training hyperparameters:
            training_params = LSTMTunable.get_training_params_from_best(study.best_params)
            model.fit(X_train, y_train, X_val, y_val, **training_params)
        """
        # Extract architecture hyperparameters
        num_layers = best_params.get('lstm_num_layers', 1)
        hidden_sizes = []
        for i in range(num_layers):
            hidden_size = best_params.get(f'lstm_hidden_size_layer_{i}', 50)
            hidden_sizes.append(hidden_size)
        
        params = {
            'input_size': input_size,
            'num_layers': num_layers,
            'hidden_sizes': hidden_sizes,
            'dropout': best_params.get('lstm_dropout', 0.0),
            'activation': best_params.get('lstm_activation', 'tanh')
        }
        return cls(trial=None, input_size=input_size, window_size=window_size, 
                   device=device, **params)
    
    @staticmethod
    def get_training_params_from_best(best_params: dict):
        """
        Extract training hyperparameters from best_params for use in fit().
        
        Args:
            best_params: Dictionary of best hyperparameters from study.best_params
            
        Returns:
            Dictionary with training hyperparameters (learning_rate, weight_decay, 
            batch_size, epochs)
            
        Example:
            training_params = LSTMTunable.get_training_params_from_best(study.best_params)
            model.fit(X_train, y_train, X_val, y_val, **training_params)
        """
        return {
            'learning_rate': best_params.get('lstm_learning_rate', 0.001),
            'weight_decay': best_params.get('lstm_weight_decay', 0.0),
            'batch_size': best_params.get('lstm_batch_size', 32),
            'epochs': best_params.get('lstm_epochs', 250)
        }
    
    @staticmethod
    def save_best_params(best_params: dict, filepath: str):
        """
        Save best hyperparameters to a JSON file.
        
        Args:
            best_params: Dictionary of best hyperparameters from study.best_params
            filepath: Path to save the parameters (e.g., 'best_params_lstm.json')
            
        Example:
            LSTMTunable.save_best_params(study.best_params, 'best_params_lstm.json')
        """
        # Filter only LSTM-related params
        lstm_params = {k: v for k, v in best_params.items() if k.startswith('lstm_')}
        # Convert to regular dict (remove 'lstm_' prefix for cleaner keys)
        clean_params = {k.replace('lstm_', ''): v for k, v in lstm_params.items()}
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(clean_params, f, indent=2)
    
    @staticmethod
    def load_best_params(filepath: str):
        """
        Load best hyperparameters from a JSON file.
        
        Args:
            filepath: Path to the saved parameters file
            
        Returns:
            Dictionary of hyperparameters ready for from_best_params()
            
        Example:
            params = LSTMTunable.load_best_params('best_params_lstm.json')
            model = LSTMTunable.from_best_params(params, input_size=17)
            training_params = LSTMTunable.get_training_params_from_best(params)
        """
        with open(filepath, 'r') as f:
            params = json.load(f)
        # Add 'lstm_' prefix back for compatibility with from_best_params
        lstm_params = {f'lstm_{k}': v for k, v in params.items()}
        return lstm_params
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Pass through each LSTM layer sequentially
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, _ = lstm_layer(x)
            # Apply activation function after each LSTM layer
            x = self.activation(x)
            # Apply dropout between layers (except after last layer)
            if i < len(self.lstm_layers) - 1 and self.dropout_layer is not None:
                x = self.dropout_layer(x)
        
        # Take the output from the last time step
        # x shape: (batch_size, window_size, hidden_size)
        # We take the last time step: (batch_size, hidden_size)
        last_output = x[:, -1, :]
        
        # Apply dropout before final layer if enabled
        if self.dropout_layer is not None:
            last_output = self.dropout_layer(last_output)
        
        # Dense layer
        output = self.fc(last_output)
        
        return output
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, trial=None, 
            device='cuda', verbose=True, **training_params):
        """
        Train the LSTM model with hyperparameters suggested by Optuna or fixed params.
        
        Args:
            X_train: Training sequences (n_samples, window_size, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation sequences (optional, required for early stopping and LR reduction)
            y_val: Validation targets (optional, required for early stopping and LR reduction)
            trial: Optuna trial object for pruning (optional)
            device: Device to train on ('cpu' or 'cuda', default: 'cuda')
            verbose: Whether to print training progress
            **training_params: Fixed training hyperparameters (learning_rate, weight_decay, 
                              batch_size, epochs) for final training with best params
        """
        self.device = device
        self.to(device)
        self.train()
        
        # Get training hyperparameters
        if trial is not None:
            learning_rate = trial.suggest_float('lstm_learning_rate', 1e-5, 1e-2, log=True)
            weight_decay = trial.suggest_float('lstm_weight_decay', 1e-6, 1e-3, log=True)
            batch_size = trial.suggest_categorical('lstm_batch_size', [16, 32, 64, 128])
            epochs = trial.suggest_int('lstm_epochs', 10, 250, step=10)
        elif training_params:
            # Use fixed training hyperparameters from kwargs (for final training)
            learning_rate = training_params.get('learning_rate', 0.001)
            weight_decay = training_params.get('weight_decay', 0.0)
            batch_size = training_params.get('batch_size', 32)
            epochs = training_params.get('epochs', 250)
        else:
            # Default values if no trial and no params provided
            learning_rate = 0.001
            weight_decay = 0.0
            batch_size = 32
            epochs = 250
        
        # Convert to tensors
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
        
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        
        # Setup learning rate scheduler (ReduceLROnPlateau)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
        )
        
        # Training loop
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        best_val_loss = float('inf')
        patience = 25  # Early stopping patience
        patience_counter = 0
        best_model_state = None  # Store best model state
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            indices = torch.randperm(n_samples, device=device)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self.evaluate(X_val, y_val, batch_size, device, criterion)
                
                # Update learning rate scheduler
                scheduler.step(val_loss)
                
                # Report to Optuna for pruning
                if trial is not None:
                    trial.report(val_loss, epoch)
                    # Prune if necessary
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                # Early stopping and save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if verbose:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {current_lr:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Load best model state if validation was used
        if X_val is not None and y_val is not None and best_model_state is not None:
            self.load_state_dict(best_model_state)
            if verbose:
                print(f"Loaded best model state (val_loss: {best_val_loss:.6f})")
    
    def predict(self, X, batch_size=32, device='cuda'):
        """
        Make predictions using the trained model.
        
        Args:
            X: Input sequences (n_samples, window_size, n_features)
            batch_size: Batch size for prediction (default: 32)
            device: Device to use for prediction ('cpu' or 'cuda', default: 'cuda')
            
        Returns:
            Predictions (n_samples, 1) as numpy array
        """
        self.eval()
        self.to(device)
        
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        X = X.to(device)
        
        predictions = []
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                X_batch = X[start_idx:end_idx]
                outputs = self.forward(X_batch)
                predictions.append(outputs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def evaluate(self, X, y, batch_size=32, device='cuda', criterion=None):
        """
        Evaluate the model on given data.
        
        Args:
            X: Input sequences (n_samples, window_size, n_features)
            y: Target values (n_samples,)
            batch_size: Batch size for evaluation (default: 32)
            device: Device to use for evaluation ('cpu' or 'cuda', default: 'cuda')
            criterion: Loss function (default: MSELoss)
            
        Returns:
            Average loss value
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.eval()
        self.to(device)
        
        # Convert to tensors
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y).unsqueeze(1)
        
        X = X.to(device)
        y = y.to(device)
        
        total_loss = 0.0
        n_samples = X.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                
                X_batch = X[start_idx:end_idx]
                y_batch = y[start_idx:end_idx]
                
                outputs = self.forward(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
        
        return total_loss / n_batches
    
    def evaluate_metrics(self, X, y, batch_size=32, device='cuda'):
        """
        Calculate evaluation metrics (MSE, RMSE, MAE, R²) as used in Nerea's implementation.
        
        Args:
            X: Input sequences (n_samples, window_size, n_features)
            y: Target values (n_samples,)
            batch_size: Batch size for evaluation (default: 32)
            device: Device to use for evaluation ('cpu' or 'cuda')
            
        Returns:
            Dictionary with metrics: {'mse': float, 'rmse': float, 'mae': float, 'r2': float}
        """
        predictions = self.predict(X, batch_size=batch_size, device=device)
        # Flatten predictions if needed
        if predictions.ndim > 1:
            predictions = predictions.flatten()
        if y.ndim > 1:
            y = y.flatten()
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }