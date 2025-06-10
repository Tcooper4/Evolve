import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple

class Optimizer:
    def __init__(self, state_dim: int, action_dim: int):
        """Initialize the optimizer with state and action dimensions."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize neural network
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
    
    def optimize_portfolio(self, state: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize portfolio weights given current state and target."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        target_tensor = torch.FloatTensor(target).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(state_tensor)
        loss = self.criterion(output, target_tensor)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return output.detach().cpu().numpy(), loss.item()
    
    def get_optimal_weights(self, state: np.ndarray) -> np.ndarray:
        """Get optimal portfolio weights for given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            weights = self.model(state_tensor)
            return weights.cpu().numpy()
    
    def update_model(self, state: np.ndarray, target: np.ndarray) -> float:
        """Update model with new data."""
        state_tensor = torch.FloatTensor(state).to(self.device)
        target_tensor = torch.FloatTensor(target).to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(state_tensor)
        loss = self.criterion(output, target_tensor)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def get_optimization_metrics(self) -> Dict[str, float]:
        """Get optimization metrics."""
        return {
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        } 