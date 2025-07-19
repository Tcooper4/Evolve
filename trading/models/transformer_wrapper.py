"""
Transformer Wrapper - Batch 17
Enhanced transformer wrapper with configurable dropout layers for training robustness
"""

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError as e:
    print("⚠️ PyTorch not available. Disabling transformer models.")
    print(f"   Missing: {e}")
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class TransformerConfig:
    """Configuration for transformer wrapper."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    layer_norm_eps: float = 1e-6
    max_seq_length: int = 1024
    use_relative_position: bool = True
    enable_dropout: bool = True

class MultiHeadAttention(nn.Module):
    """Multi-head attention with configurable dropout."""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create MultiHeadAttention.")
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with attention dropout.
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor with dropout applied
        """
        batch_size, seq_len, _ = query.size()
        
        # Linear projections and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention dropout
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        # Apply output dropout
        output = self.output_dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(query + output)
        
        return output

class FeedForward(nn.Module):
    """Feed-forward network with configurable dropout."""
    
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 dropout_rate: float = 0.1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create FeedForward.")
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout."""
        residual = x
        x = F.relu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = self.layer_norm(residual + x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with configurable dropout after attention."""
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int,
                 dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.1,
                 enable_dropout: bool = True):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create TransformerBlock.")
        super().__init__()
        self.enable_dropout = enable_dropout
        
        # Attention layer
        self.attention = MultiHeadAttention(
            d_model, n_heads, dropout_rate, attention_dropout_rate
        )
        
        # Dropout after attention block
        self.attention_dropout = nn.Dropout(dropout_rate) if enable_dropout else nn.Identity()
        
        # Feed-forward layer
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        
        # Dropout after feed-forward block
        self.ff_dropout = nn.Dropout(dropout_rate) if enable_dropout else nn.Identity()
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with dropout after each block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor with dropout applied
        """
        # Self-attention with dropout
        attn_output = self.attention(x, x, x, mask)
        attn_output = self.attention_dropout(attn_output)
        
        # Feed-forward with dropout
        ff_output = self.feed_forward(attn_output)
        ff_output = self.ff_dropout(ff_output)
        
        return ff_output

class TransformerWrapper(nn.Module):
    """
    Enhanced transformer wrapper with configurable dropout layers.
    
    Features:
    - Configurable dropout after each attention block
    - Training/evaluation mode dropout control
    - Layer-wise dropout statistics
    - Gradient monitoring
    """
    
    def __init__(self, config: TransformerConfig):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Cannot create TransformerWrapper.")
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Linear(config.d_model, config.d_model)
        self.position_encoding = self._create_position_encoding()
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout_rate,
                config.attention_dropout_rate,
                config.enable_dropout
            )
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        
        # Final dropout
        self.final_dropout = nn.Dropout(config.dropout_rate) if config.enable_dropout else nn.Identity()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Statistics tracking
        self.dropout_stats = {
            'attention_dropout_rate': 0.0,
            'ff_dropout_rate': 0.0,
            'final_dropout_rate': 0.0
        }
        
        logger.info(f"TransformerWrapper initialized with {config.n_layers} layers")
        logger.info(f"Dropout enabled: {config.enable_dropout}, rate: {config.dropout_rate}")
    
    def _create_position_encoding(self) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(self.config.max_seq_length, self.config.d_model)
        position = torch.arange(0, self.config.max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2).float() *
                           -(math.log(10000.0) / self.config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass with configurable dropout.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Output tensor or tuple with attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Add positional encoding
        if seq_len <= self.config.max_seq_length:
            x = x + self.position_encoding[:, :seq_len].to(x.device)
        
        # Apply embedding projection
        x = self.embedding(x)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Pass through transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if return_attention_weights:
                # Custom forward to get attention weights
                attn_output, attn_weights = self._forward_with_attention_weights(block, x, mask)
                attention_weights.append(attn_weights)
                x = attn_output
            else:
                x = block(x, mask)
            
            # Update dropout statistics
            if self.training and self.config.enable_dropout:
                self._update_dropout_stats(i, x)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Final dropout
        x = self.final_dropout(x)
        
        if return_attention_weights:
            return x, attention_weights
        else:
            return x
    
    def _forward_with_attention_weights(self, 
                                      block: TransformerBlock,
                                      x: torch.Tensor, 
                                      mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that returns attention weights."""
        # This is a simplified version - in practice you'd modify the attention layer
        # to return weights. For now, we'll just return the output and dummy weights
        output = block(x, mask)
        dummy_weights = torch.zeros(x.size(0), block.attention.n_heads, x.size(1), x.size(1))
        return output, dummy_weights
    
    def _update_dropout_stats(self, layer_idx: int, x: torch.Tensor):
        """Update dropout statistics for monitoring."""
        if self.training:
            # Calculate dropout rate (simplified)
            dropout_rate = (x == 0).float().mean().item()
            self.dropout_stats[f'layer_{layer_idx}_dropout_rate'] = dropout_rate
    
    def get_dropout_stats(self) -> Dict[str, float]:
        """Get current dropout statistics."""
        return self.dropout_stats.copy()
    
    def set_dropout_rate(self, rate: float):
        """Set dropout rate for all layers."""
        self.config.dropout_rate = rate
        self.config.attention_dropout_rate = rate
        
        # Update dropout layers
        for block in self.transformer_blocks:
            block.attention.attention_dropout.p = rate
            block.attention.output_dropout.p = rate
            block.attention_dropout.p = rate
            block.ff_dropout.p = rate
            block.feed_forward.dropout.p = rate
        
        self.final_dropout.p = rate
        logger.info(f"Updated dropout rate to {rate}")
    
    def enable_dropout(self, enable: bool = True):
        """Enable or disable dropout."""
        self.config.enable_dropout = enable
        
        # Update dropout layers
        for block in self.transformer_blocks:
            block.enable_dropout = enable
            block.attention_dropout = nn.Dropout(self.config.dropout_rate) if enable else nn.Identity()
            block.ff_dropout = nn.Dropout(self.config.dropout_rate) if enable else nn.Identity()
        
        self.final_dropout = nn.Dropout(self.config.dropout_rate) if enable else nn.Identity()
        
        mode = "enabled" if enable else "disabled"
        logger.info(f"Dropout {mode}")
    
    def get_gradient_norm(self) -> float:
        """Get the L2 norm of gradients."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

def create_transformer_wrapper(config: Optional[TransformerConfig] = None) -> TransformerWrapper:
    """Factory function to create a transformer wrapper."""
    if config is None:
        config = TransformerConfig()
    return TransformerWrapper(config) 