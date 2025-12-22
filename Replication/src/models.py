"""
Model implementations based on architectures from older paper.

This module contains:
- RandomForest: scikit-learn Random Forest regressor
- XGBoost: XGBoost gradient boosting regressor
- LSTM: PyTorch LSTM model for time series prediction
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


class RandomForest:
    """
    Random Forest model using scikit-learn.
    
    Uses all available CPU threads for parallel training.
    
    Architecture from Nerea/04_Modelo/RF.ipynb:
    - n_estimators=100
    - max_depth=10
    - random_state=42
    """
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of the tree (default: 10)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1  # Use all available CPU threads
        )
    
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


class XGBoost:
    """
    XGBoost model using the xgboost library.
    
    Uses all available CPU threads for parallel training.
    
    Default parameters:
    - n_estimators=100
    - max_depth=6
    - learning_rate=0.1
    - random_state=42
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, 
                 subsample=1.0, colsample_bytree=1.0, random_state=42):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds (default: 100)
            max_depth: Maximum depth of trees (default: 6)
            learning_rate: Learning rate (default: 0.1)
            subsample: Subsample ratio of training instances (default: 1.0)
            colsample_bytree: Subsample ratio of columns when constructing each tree (default: 1.0)
            random_state: Random seed for reproducibility (default: 42)
        """
        
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,  # Use all available CPU threads
            tree_method='hist'  # Use histogram-based algorithm for efficiency
        )
    
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


class LSTM(nn.Module):
    """
    LSTM model using PyTorch.
    
    Architecture from Nerea/04_Modelo/LSTMs.ipynb:
    - LSTM layer with 50 hidden units, tanh activation
    - Dense output layer with 1 unit
    - Input shape: (batch_size, window_size, n_features)
    - Window size: 6
    - Optimizer: Adam with learning_rate=0.001
    - Loss: MSE
    - Training: 20 epochs, batch_size=32
    """
    
    def __init__(self, input_size, hidden_size=50, num_layers=1, window_size=6):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features per time step
            hidden_size: Number of hidden units in LSTM (default: 50)
            num_layers: Number of LSTM layers (default: 1)
            window_size: Size of the input window (default: 6)
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        
        # LSTM layer with tanh activation (default)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Dense output layer
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        # lstm_out shape: (batch_size, window_size, hidden_size)
        # We take the last time step: (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Dense layer
        output = self.fc(last_output)
        
        return output
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs=250, batch_size=32, learning_rate=0.001, 
            device='cuda', verbose=True):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences (n_samples, window_size, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation sequences (optional, required for early stopping and LR reduction)
            y_val: Validation targets (optional, required for early stopping and LR reduction)
            epochs: Maximum number of training epochs (default: 250)
            batch_size: Batch size for training (default: 32)
            learning_rate: Learning rate for Adam optimizer (default: 0.001)
            device: Device to train on ('cpu' or 'cuda')
            verbose: Whether to print training progress
        """
        self.to(device)
        self.train()
        
        # Convert to tensors
        if not isinstance(X_train, torch.Tensor):
            X_train = torch.FloatTensor(X_train)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.FloatTensor(y_train).unsqueeze(1)
        
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Setup learning rate scheduler (ReduceLROnPlateau)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
        )
        
        # Training loop
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience = 25
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
            device: Device to use for prediction ('cpu' or 'cuda')
            
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
            device: Device to use for evaluation ('cpu' or 'cuda')
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
