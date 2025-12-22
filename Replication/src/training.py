"""
Training script for all models.

This module provides functions to train:
- Basic models (from models.py): RandomForest, XGBoost, LSTM (no hyperparameter tuning)
- Tunable models (from models_finetunned.py): RandomForestTunable, XGBoostTunable, LSTMTunable (with Optuna hyperparameter tuning)

Data loading should be implemented separately and passed to these functions.
"""

import os
import json
from pathlib import Path
import numpy as np
import optuna
from optuna.pruners import MedianPruner
import torch
import joblib
from sklearn.metrics import mean_squared_error

# Basic models (no hyperparameter tuning)
from .models import RandomForest, XGBoost, LSTM

# Tunable models (with hyperparameter tuning)
from .models_finetunned import (
    RandomForestTunable,
    XGBoostTunable,
    LSTMTunable
)


# ============================================================================
# BASIC MODELS TRAINING (from models.py)
# ============================================================================

def train_random_forest(X_train, y_train, X_test=None, y_test=None,
                       n_estimators=100, max_depth=10, random_state=42,
                       save_path=None):
    """
    Train Random Forest model (basic version, no hyperparameter tuning).
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_test: Test features (optional, for evaluation)
        y_test: Test targets (optional, for evaluation)
        n_estimators: Number of trees (default: 100)
        max_depth: Maximum depth of trees (default: 10)
        random_state: Random seed (default: 42)
        save_path: Path to save the trained model (default: 'models/rf_basic_model.joblib')
        
    Returns:
        Trained RandomForest model
    """
    print("Training Random Forest (basic)...")
    model = RandomForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    if X_test is not None and y_test is not None:
        # Use RMSE as primary metric (better for regression optimization)
        metrics = model.evaluate_metrics(X_test, y_test)
        print(f"Random Forest Test Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} (primary)")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Set default save path if not provided
    if save_path is None:
        os.makedirs('models', exist_ok=True)
        save_path = 'models/rf_basic_model.joblib'
    
    # Save model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    joblib.dump(model.model, save_path)
    print(f"Model saved to: {save_path}")
    
    return model


def train_xgboost(X_train, y_train, X_test=None, y_test=None,
                  n_estimators=100, max_depth=6, learning_rate=0.1,
                  subsample=1.0, colsample_bytree=1.0, random_state=42,
                  save_path=None):
    """
    Train XGBoost model (basic version, no hyperparameter tuning).
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_test: Test features (optional, for evaluation)
        y_test: Test targets (optional, for evaluation)
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum depth of trees (default: 6)
        learning_rate: Learning rate (default: 0.1)
        subsample: Subsample ratio (default: 1.0)
        colsample_bytree: Column subsample ratio (default: 1.0)
        random_state: Random seed (default: 42)
        save_path: Path to save the trained model (default: 'models/xgb_basic_model.json')
        
    Returns:
        Trained XGBoost model
    """
    print("Training XGBoost (basic)...")
    model = XGBoost(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    
    if X_test is not None and y_test is not None:
        # Use RMSE as primary metric (better for regression optimization)
        metrics = model.evaluate_metrics(X_test, y_test)
        print(f"XGBoost Test Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} (primary)")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Set default save path if not provided
    if save_path is None:
        os.makedirs('models', exist_ok=True)
        save_path = 'models/xgb_basic_model.json'
    
    # Save model using XGBoost's built-in save method
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    model.model.save_model(save_path)
    print(f"Model saved to: {save_path}")
    
    return model


def train_lstm(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None,
               input_size=None, hidden_size=50, num_layers=1, window_size=6,
               epochs=250, batch_size=32, learning_rate=0.001,
               device='cuda', save_path=None):
    """
    Train LSTM model (basic version, no hyperparameter tuning).
    
    Args:
        X_train: Training sequences (n_samples, window_size, n_features)
        y_train: Training targets (n_samples,)
        X_val: Validation sequences (optional, for early stopping)
        y_val: Validation targets (optional, for early stopping)
        X_test: Test sequences (optional, for evaluation)
        y_test: Test targets (optional, for evaluation)
        input_size: Number of input features per time step (inferred from X_train if None)
        hidden_size: Number of hidden units (default: 50)
        num_layers: Number of LSTM layers (default: 1)
        window_size: Size of input window (default: 6)
        epochs: Maximum number of epochs (default: 250)
        batch_size: Batch size (default: 32)
        learning_rate: Learning rate (default: 0.001)
        device: Device to train on ('cpu' or 'cuda', default: 'cuda')
        save_path: Path to save the trained model (optional)
        
    Returns:
        Trained LSTM model
    """
    print("Training LSTM (basic)...")
    
    # Infer input_size from data if not provided
    if input_size is None:
        if len(X_train.shape) == 3:
            input_size = X_train.shape[2]
        else:
            raise ValueError("Cannot infer input_size from X_train. Please provide input_size.")
    
    model = LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        window_size=window_size
    )
    
    model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device
    )
    
    if X_test is not None and y_test is not None:
        # Use RMSE as primary metric (better for regression optimization)
        metrics = model.evaluate_metrics(X_test, y_test, device=device)
        print(f"LSTM Test Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} (primary)")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Set default save path if not provided
    if save_path is None:
        os.makedirs('models', exist_ok=True)
        save_path = 'models/lstm_basic_model.pt'
    
    # Save model
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'window_size': window_size
    }, save_path)
    print(f"Model saved to: {save_path}")
    
    return model


# ============================================================================
# TUNABLE MODELS TRAINING (from models_finetunned.py)
# ============================================================================

def train_random_forest_tunable(X_train, y_train, X_val, y_val,
                                n_trials=100, study_name=None, storage=None,
                                direction='maximize', pruner=None,
                                save_params_path=None,
                                X_test=None, y_test=None, save_model_path=None):
    """
    Train Random Forest with Optuna hyperparameter tuning.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_val: Validation features (for hyperparameter tuning)
        y_val: Validation targets (for hyperparameter tuning)
        n_trials: Number of Optuna trials (default: 100)
        study_name: Name for the Optuna study (optional)
        storage: Storage backend for Optuna study (optional, e.g., 'sqlite:///study.db')
        direction: Optimization direction ('maximize' or 'minimize', default: 'maximize')
        pruner: Optuna pruner (default: MedianPruner)
        save_params_path: Path to save best hyperparameters (default: 'models/best_params_rf.json')
        X_test: Test features (optional, for final evaluation)
        y_test: Test targets (optional, for final evaluation)
        save_model_path: Path to save final trained model (default: 'models/rf_tuned_model.joblib')
        
    Returns:
        Tuple of (trained model, Optuna study)
    """
    print(f"Training Random Forest with hyperparameter tuning ({n_trials} trials)...")
    
    def objective(trial):
        model = RandomForestTunable(trial)
        model.fit(X_train, y_train)
        # Optimize for RMSE instead of R² (better for regression optimization)
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        return rmse
    
    # Create study - optimize to minimize RMSE
    if pruner is None:
        pruner = MedianPruner()
    
    # Override direction to 'minimize' for RMSE (unless explicitly set)
    opt_direction = direction if direction in ['maximize', 'minimize'] else 'minimize'
    
    study = optuna.create_study(
        direction=opt_direction,
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Set default save paths if not provided
    if save_params_path is None:
        os.makedirs('models', exist_ok=True)
        save_params_path = 'models/best_params_rf.json'
    if save_model_path is None:
        os.makedirs('models', exist_ok=True)
        save_model_path = 'models/rf_tuned_model.joblib'
    
    # Save best parameters
    os.makedirs(os.path.dirname(save_params_path) if os.path.dirname(save_params_path) else '.', exist_ok=True)
    RandomForestTunable.save_best_params(study.best_params, save_params_path)
    print(f"Best parameters saved to: {save_params_path}")
    print(f"Best validation RMSE: {study.best_value:.4f}")
    
    # Train final model with best parameters
    print("Training final model with best hyperparameters...")
    final_model = RandomForestTunable.from_best_params(study.best_params)
    final_model.fit(X_train, y_train)  # Use full training set
    
    if X_test is not None and y_test is not None:
        # Use RMSE as primary metric (better for regression optimization)
        metrics = final_model.evaluate_metrics(X_test, y_test)
        print(f"Final Random Forest Test Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} (primary)")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else '.', exist_ok=True)
    joblib.dump(final_model.model, save_model_path)
    print(f"Final model saved to: {save_model_path}")
    
    return final_model, study


def train_xgboost_tunable(X_train, y_train, X_val, y_val,
                          n_trials=100, study_name=None, storage=None,
                          direction='maximize', pruner=None,
                          save_params_path=None,
                          X_test=None, y_test=None, save_model_path=None):
    """
    Train XGBoost with Optuna hyperparameter tuning.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training targets (n_samples,)
        X_val: Validation features (for hyperparameter tuning)
        y_val: Validation targets (for hyperparameter tuning)
        n_trials: Number of Optuna trials (default: 100)
        study_name: Name for the Optuna study (optional)
        storage: Storage backend for Optuna study (optional)
        direction: Optimization direction ('maximize' or 'minimize', default: 'maximize')
        pruner: Optuna pruner (default: MedianPruner)
        save_params_path: Path to save best hyperparameters (default: 'models/best_params_xgb.json')
        X_test: Test features (optional, for final evaluation)
        y_test: Test targets (optional, for final evaluation)
        save_model_path: Path to save final trained model (default: 'models/xgb_tuned_model.json')
        
    Returns:
        Tuple of (trained model, Optuna study)
    """
    print(f"Training XGBoost with hyperparameter tuning ({n_trials} trials)...")
    
    def objective(trial):
        model = XGBoostTunable(trial)
        model.fit(X_train, y_train)
        # Optimize for RMSE instead of R² (better for regression optimization)
        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        return rmse
    
    # Create study - optimize to minimize RMSE
    if pruner is None:
        pruner = MedianPruner()
    
    # Override direction to 'minimize' for RMSE (unless explicitly set)
    opt_direction = direction if direction in ['maximize', 'minimize'] else 'minimize'
    
    study = optuna.create_study(
        direction=opt_direction,
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Set default save paths if not provided
    if save_params_path is None:
        os.makedirs('models', exist_ok=True)
        save_params_path = 'models/best_params_xgb.json'
    if save_model_path is None:
        os.makedirs('models', exist_ok=True)
        save_model_path = 'models/xgb_tuned_model.json'
    
    # Save best parameters
    os.makedirs(os.path.dirname(save_params_path) if os.path.dirname(save_params_path) else '.', exist_ok=True)
    XGBoostTunable.save_best_params(study.best_params, save_params_path)
    print(f"Best parameters saved to: {save_params_path}")
    print(f"Best validation RMSE: {study.best_value:.4f}")
    
    # Train final model with best parameters
    print("Training final model with best hyperparameters...")
    final_model = XGBoostTunable.from_best_params(study.best_params)
    final_model.fit(X_train, y_train)  # Use full training set
    
    if X_test is not None and y_test is not None:
        # Use RMSE as primary metric (better for regression optimization)
        metrics = final_model.evaluate_metrics(X_test, y_test)
        print(f"Final XGBoost Test Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} (primary)")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Save model using XGBoost's built-in save method
    os.makedirs(os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else '.', exist_ok=True)
    final_model.model.save_model(save_model_path)
    print(f"Final model saved to: {save_model_path}")
    
    return final_model, study


def train_lstm_tunable(X_train, y_train, X_val, y_val,
                       input_size=None, window_size=6,
                       n_trials=100, study_name=None, storage=None,
                       direction='minimize', pruner=None,
                       save_params_path=None,
                       X_test=None, y_test=None, save_model_path=None,
                       device='cuda'):
    """
    Train LSTM with Optuna hyperparameter tuning.
    
    Args:
        X_train: Training sequences (n_samples, window_size, n_features)
        y_train: Training targets (n_samples,)
        X_val: Validation sequences (for hyperparameter tuning)
        y_val: Validation targets (for hyperparameter tuning)
        input_size: Number of input features per time step (inferred from X_train if None)
        window_size: Size of input window (default: 6)
        n_trials: Number of Optuna trials (default: 100)
        study_name: Name for the Optuna study (optional)
        storage: Storage backend for Optuna study (optional)
        direction: Optimization direction ('maximize' or 'minimize', default: 'minimize' for loss)
        pruner: Optuna pruner (default: MedianPruner)
        save_params_path: Path to save best hyperparameters (default: 'models/best_params_lstm.json')
        X_test: Test sequences (optional, for final evaluation)
        y_test: Test targets (optional, for final evaluation)
        save_model_path: Path to save final trained model (default: 'models/lstm_tuned_model.pt')
        device: Device to train on ('cpu' or 'cuda', default: 'cuda')
        
    Returns:
        Tuple of (trained model, Optuna study)
    """
    print(f"Training LSTM with hyperparameter tuning ({n_trials} trials)...")
    
    # Infer input_size from data if not provided
    if input_size is None:
        if len(X_train.shape) == 3:
            input_size = X_train.shape[2]
        else:
            raise ValueError("Cannot infer input_size from X_train. Please provide input_size.")
    
    def objective(trial):
        model = LSTMTunable(trial, input_size=input_size, window_size=window_size)
        
        # Get training hyperparameters from trial
        learning_rate = trial.suggest_float('lstm_learning_rate', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('lstm_weight_decay', 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical('lstm_batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('lstm_epochs', 50, 250)
        
        model.fit(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            trial=trial,
            device=device,
            verbose=False,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs
        )
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            if not isinstance(X_val, torch.Tensor):
                X_val_tensor = torch.FloatTensor(X_val).to(device)
            else:
                X_val_tensor = X_val.to(device)
            
            predictions = model(X_val_tensor)
            if not isinstance(y_val, torch.Tensor):
                y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(device)
            else:
                y_val_tensor = y_val.unsqueeze(1).to(device)
            
            # Optimize for RMSE instead of MSE (consistent with other models)
            # RMSE = sqrt(MSE), so same ranking but more interpretable
            mse_loss = torch.nn.functional.mse_loss(predictions, y_val_tensor)
            rmse = torch.sqrt(mse_loss)
            return rmse.item()
    
    # Create study
    if pruner is None:
        pruner = MedianPruner()
    
    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        pruner=pruner,
        load_if_exists=True
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Set default save paths if not provided
    if save_params_path is None:
        os.makedirs('models', exist_ok=True)
        save_params_path = 'models/best_params_lstm.json'
    if save_model_path is None:
        os.makedirs('models', exist_ok=True)
        save_model_path = 'models/lstm_tuned_model.pt'
    
    # Save best parameters
    os.makedirs(os.path.dirname(save_params_path) if os.path.dirname(save_params_path) else '.', exist_ok=True)
    LSTMTunable.save_best_params(study.best_params, save_params_path)
    print(f"Best parameters saved to: {save_params_path}")
    print(f"Best validation RMSE: {study.best_value:.4f}")
    
    # Train final model with best parameters
    print("Training final model with best hyperparameters...")
    final_model = LSTMTunable.from_best_params(study.best_params, input_size=input_size, window_size=window_size)
    training_params = LSTMTunable.get_training_params_from_best(study.best_params)
    
    final_model.fit(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        device=device,
        verbose=True,
        **training_params
    )
    
    if X_test is not None and y_test is not None:
        # Use RMSE as primary metric (better for regression optimization)
        metrics = final_model.evaluate_metrics(X_test, y_test, device=device)
        print(f"Final LSTM Test Metrics:")
        print(f"  RMSE: {metrics['rmse']:.4f} (primary)")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Save model (best model state is already loaded by fit method)
    os.makedirs(os.path.dirname(save_model_path) if os.path.dirname(save_model_path) else '.', exist_ok=True)
    
    # Extract hidden sizes for all layers
    num_layers = study.best_params.get('lstm_num_layers', 1)
    hidden_sizes = [study.best_params.get(f'lstm_hidden_size_layer_{i}', 50) for i in range(num_layers)]
    
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_size': input_size,
        'window_size': window_size,
        'num_layers': num_layers,
        'hidden_sizes': hidden_sizes,
        'activation': study.best_params.get('lstm_activation', 'tanh'),
        'dropout': study.best_params.get('lstm_dropout', 0.0),
        'best_params': study.best_params  # Keep full params for reference
    }, save_model_path)
    print(f"Final model saved to: {save_model_path}")
    
    return final_model, study


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def train_all_basic_models(X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None,
                           lstm_input_size=None, lstm_window_size=6, device='cuda',
                           save_dir='models'):
    """
    Train all basic models (RandomForest, XGBoost, LSTM) without hyperparameter tuning.
    
    Args:
        X_train: Training features/sequences
        y_train: Training targets
        X_val: Validation features/sequences (optional, for LSTM early stopping)
        y_val: Validation targets (optional, for LSTM early stopping)
        X_test: Test features/sequences (optional, for evaluation)
        y_test: Test targets (optional, for evaluation)
        lstm_input_size: Input size for LSTM (inferred if None)
        lstm_window_size: Window size for LSTM (default: 6)
        device: Device for LSTM training (default: 'cuda')
        save_dir: Directory to save models (default: 'models')
        
    Returns:
        Dictionary with trained models: {'rf': model, 'xgb': model, 'lstm': model}
    """
    models = {}
    
    # Train Random Forest
    rf_save_path = os.path.join(save_dir, 'rf_basic_model.joblib')
    models['rf'] = train_random_forest(X_train, y_train, X_test, y_test, save_path=rf_save_path)
    
    # Train XGBoost
    xgb_save_path = os.path.join(save_dir, 'xgb_basic_model.json')
    models['xgb'] = train_xgboost(X_train, y_train, X_test, y_test, save_path=xgb_save_path)
    
    # Train LSTM (requires sequences)
    # Note: X_train for LSTM should be (n_samples, window_size, n_features)
    lstm_save_path = os.path.join(save_dir, 'lstm_basic_model.pt')
    models['lstm'] = train_lstm(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_size=lstm_input_size, window_size=lstm_window_size,
        device=device, save_path=lstm_save_path
    )
    
    return models


def train_all_tunable_models(X_train, y_train, X_val, y_val, X_test=None, y_test=None,
                              lstm_input_size=None, lstm_window_size=6,
                              n_trials=100, device='cuda', save_dir='models'):
    """
    Train all tunable models with Optuna hyperparameter tuning.
    
    Args:
        X_train: Training features/sequences
        y_train: Training targets
        X_val: Validation features/sequences (required for hyperparameter tuning)
        y_val: Validation targets (required for hyperparameter tuning)
        X_test: Test features/sequences (optional, for final evaluation)
        y_test: Test targets (optional, for final evaluation)
        lstm_input_size: Input size for LSTM (inferred if None)
        lstm_window_size: Window size for LSTM (default: 6)
        n_trials: Number of Optuna trials per model (default: 100)
        device: Device for LSTM training (default: 'cuda')
        save_dir: Directory to save models and parameters (default: 'models')
        
    Returns:
        Dictionary with trained models and studies:
        {
            'rf': (model, study),
            'xgb': (model, study),
            'lstm': (model, study)
        }
    """
    results = {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Train Random Forest with tuning
    rf_params_path = os.path.join(save_dir, 'best_params_rf.json')
    rf_model_path = os.path.join(save_dir, 'rf_tuned_model.joblib')
    results['rf'] = train_random_forest_tunable(
        X_train, y_train, X_val, y_val,
        n_trials=n_trials,
        save_params_path=rf_params_path,
        X_test=X_test, y_test=y_test,
        save_model_path=rf_model_path
    )
    
    # Train XGBoost with tuning
    xgb_params_path = os.path.join(save_dir, 'best_params_xgb.json')
    xgb_model_path = os.path.join(save_dir, 'xgb_tuned_model.json')
    results['xgb'] = train_xgboost_tunable(
        X_train, y_train, X_val, y_val,
        n_trials=n_trials,
        save_params_path=xgb_params_path,
        X_test=X_test, y_test=y_test,
        save_model_path=xgb_model_path
    )
    
    # Train LSTM with tuning (requires sequences)
    # Note: X_train for LSTM should be (n_samples, window_size, n_features)
    lstm_params_path = os.path.join(save_dir, 'best_params_lstm.json')
    lstm_model_path = os.path.join(save_dir, 'lstm_tuned_model.pt')
    results['lstm'] = train_lstm_tunable(
        X_train, y_train, X_val, y_val,
        input_size=lstm_input_size, window_size=lstm_window_size,
        n_trials=n_trials, device=device,
        save_params_path=lstm_params_path,
        X_test=X_test, y_test=y_test,
        save_model_path=lstm_model_path
    )
    
    return results
