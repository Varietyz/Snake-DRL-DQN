# ai/simple_nn.py
import numpy as np
import json
from pathlib import Path
from typing import Union, List, Optional
from enum import Enum

class Activation(Enum):
    RELU = 'relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    LINEAR = 'linear'

class Optimizer(Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    RMSPROP = 'rmsprop'

class SimpleNN:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [64],
        output_size: int = 1,
        learning_rate: float = 0.001,
        activation: Activation = Activation.RELU,
        output_activation: Activation = Activation.LINEAR,
        optimizer: Optimizer = Optimizer.ADAM,
        l2_reg: float = 0.001,
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        clip_value: float = 5.0,
        seed: Optional[int] = None
    ):
        if seed is not None:
            np.random.seed(seed)

        self.architecture = {
            'input_size': input_size,
            'hidden_sizes': hidden_sizes,
            'output_size': output_size,
            'learning_rate': learning_rate,
            'activation': activation.value,
            'output_activation': output_activation.value,
            'optimizer': optimizer.value,
            'l2_reg': l2_reg,
            'dropout_rate': dropout_rate,
            'batch_norm': batch_norm,
            'clip_value': clip_value
        }
        
        # Initialize layers
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes)-1):
            # He initialization for ReLU, Xavier for tanh/sigmoid
            if activation == Activation.RELU:
                std = np.sqrt(2.0 / layer_sizes[i])
            else:
                std = np.sqrt(1.0 / layer_sizes[i])
                
            weights = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std
            biases = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'W': weights, 'b': biases})
        
        # Training parameters
        self.lr = learning_rate
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.clip_value = clip_value
        self.activation = activation
        self.output_activation = output_activation
        self.optimizer = optimizer
        
        # Initialize optimizer parameters
        if optimizer == Optimizer.ADAM:
            self.t = 0
            self.m = [{'W': np.zeros_like(layer['W']), 'b': np.zeros_like(layer['b'])} 
                     for layer in self.layers]
            self.v = [{'W': np.zeros_like(layer['W']), 'b': np.zeros_like(layer['b'])} 
                     for layer in self.layers]
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps = 1e-8
        
        # Batch norm parameters
        if batch_norm:
            self.gamma = [np.ones((1, size)) for size in layer_sizes[1:-1]]
            self.beta = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
            self.running_mean = [np.zeros((1, size)) for size in layer_sizes[1:-1]]
            self.running_var = [np.ones((1, size)) for size in layer_sizes[1:-1]]
            self.bn_eps = 1e-5
            self.momentum = 0.9
    
    def _activate(self, x: np.ndarray, activation: Activation) -> np.ndarray:
        """Apply activation function"""
        if activation == Activation.RELU:
            return np.maximum(0, x)
        elif activation == Activation.TANH:
            return np.tanh(x)
        elif activation == Activation.SIGMOID:
            return 1 / (1 + np.exp(-x))
        elif activation == Activation.LINEAR:
            return x
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _activate_derivative(self, x: np.ndarray, activation: Activation) -> np.ndarray:
        """Derivative of activation function"""
        if activation == Activation.RELU:
            return (x > 0).astype(float)
        elif activation == Activation.TANH:
            return 1 - np.tanh(x)**2
        elif activation == Activation.SIGMOID:
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif activation == Activation.LINEAR:
            return np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _apply_dropout(self, a: np.ndarray, layer_index: int) -> np.ndarray:
        if self.dropout_rate <= 0 or layer_index >= len(self.layers) - 1:
            return a
        mask = self.cache.get(f'D{layer_index+1}', np.ones_like(a))
        return a * mask / (1 - self.dropout_rate)

    def _clean_input(self, data):
        # Example: ensure data is a numpy array with correct dtype
        import numpy as np
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        return data

    def forward(self, X: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through the network"""
        # Clean and validate input
        X = self._clean_input(X)
        self.cache = {'A0': X}
                
        for i, layer in enumerate(self.layers[:-1]):
            z = np.dot(self.cache[f'A{i}'], layer['W']) + layer['b']
            
            if self.batch_norm and i < len(self.layers) - 1:  # keep as is
                if training:
                    batch_mean = np.mean(z, axis=0, keepdims=True)
                    batch_var = np.var(z, axis=0, keepdims=True)
                    self.running_mean[i] = self.momentum * self.running_mean[i] + (1 - self.momentum) * batch_mean
                    self.running_var[i] = self.momentum * self.running_var[i] + (1 - self.momentum) * batch_var
                else:
                    batch_mean = self.running_mean[i]
                    batch_var = self.running_var[i]

                z = (z - batch_mean) / np.sqrt(batch_var + self.bn_eps)
                z = self.gamma[i] * z + self.beta[i]

            a = self._activate(z, self.activation)

            # Fix: Generate mask once here, do NOT generate mask inside _apply_dropout
            if training and self.dropout_rate > 0 and i < len(self.layers) - 1:  # include all hidden layers
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
                self.cache[f'D{i+1}'] = mask
                a = a * mask / (1 - self.dropout_rate)

            self.cache[f'Z{i+1}'] = z
            self.cache[f'A{i+1}'] = a

        
        # Output layer
        output_layer = self.layers[-1]
        z_out = np.dot(self.cache[f'A{len(self.layers)-1}'], output_layer['W']) + output_layer['b']
        self.cache[f'Z{len(self.layers)}'] = z_out  # Cache pre-activation for output layer
        output = self._activate(z_out, self.output_activation)

        
        return output
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> None:
        """Modular backward pass"""
        X, y, output = self._clean_input(X), self._clean_input(y), self._clean_input(output)
        m = X.shape[0]
        grads = [{} for _ in range(len(self.layers))]

        dZ = self._compute_output_delta(output, y)
        grads[-1]['W'], grads[-1]['b'] = self._gradients(dZ, self.cache[f'A{len(self.layers)-1}'], m)

        for i in reversed(range(len(self.layers)-1)):
            dA = np.dot(dZ, self.layers[i+1]['W'].T)
            if self.dropout_rate > 0 and i < len(self.layers) - 1:
                dA *= self.cache.get(f'D{i+1}', 1.0) / (1 - self.dropout_rate)
            dZ = dA * self._activate_derivative(self.cache[f'Z{i+1}'], self.activation)
            dZ = np.clip(dZ, -self.clip_value, self.clip_value)
            grads[i]['W'], grads[i]['b'] = self._gradients(dZ, self.cache[f'A{i}'], m)

        self._update_weights(grads)

    def _compute_output_delta(self, y_pred, y_true):
        """Return dZ for output layer"""
        if self.output_activation == Activation.LINEAR:
            return np.clip(y_pred - y_true, -self.clip_value, self.clip_value)
        dA = y_pred - y_true
        return dA * self._activate_derivative(self.cache[f'Z{len(self.layers)}'], self.output_activation)

    def _gradients(self, dZ, A_prev, m):
        """Return gradients for a single layer"""
        dW = np.dot(A_prev.T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        return dW, db

    
    def _batch_norm_backward(self, dZ, Z, batch_mean, batch_var, layer_idx):
        """Backward pass through batch normalization"""
        std = np.sqrt(batch_var + self.bn_eps)
        Z_centered = Z - batch_mean
        inv_std = 1.0 / std

        dstd = np.sum(dZ * Z_centered * (-1 / std**2), axis=0, keepdims=True)
        dmean = np.sum(dZ * (-inv_std), axis=0, keepdims=True) + dstd * np.mean(-2 * Z_centered, axis=0, keepdims=True)
        
        dZ = dZ * inv_std + dstd * 2 * Z_centered / Z.shape[0] + dmean / Z.shape[0]
        return dZ
    
    def _update_weights(self, grads):
        """Update weights using selected optimizer"""
        for i, (layer, grad) in enumerate(zip(self.layers, grads)):
            # L2 regularization
            grad['W'] += self.l2_reg * layer['W']
            
            if self.optimizer == Optimizer.SGD:
                layer['W'] -= self.lr * grad['W']
                layer['b'] -= self.lr * grad['b']
            elif self.optimizer == Optimizer.ADAM:
                self.t += 1
                # Update momentum estimates
                self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * grad['W']
                self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * grad['b']
                
                # Update RMS estimates
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * (grad['W']**2)
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * (grad['b']**2)
                
                # Bias correction
                m_hat_W = self.m[i]['W'] / (1 - self.beta1**self.t)
                m_hat_b = self.m[i]['b'] / (1 - self.beta1**self.t)
                v_hat_W = self.v[i]['W'] / (1 - self.beta2**self.t)
                v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)
                
                # Update weights
                layer['W'] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
                layer['b'] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
        
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to file"""
        model_data = {
            'architecture': self.architecture,
            'weights': [{'W': layer['W'].tolist(), 'b': layer['b'].tolist()} 
                        for layer in self.layers],
            'optimizer_state': None
        }
        
        if self.optimizer == Optimizer.ADAM:
            model_data['optimizer_state'] = {
                't': self.t,
                'm': [{'W': m['W'].tolist(), 'b': m['b'].tolist()} for m in self.m],
                'v': [{'W': v['W'].tolist(), 'b': v['b'].tolist()} for v in self.v]
            }
        
        if self.batch_norm:
            model_data['batch_norm'] = {
                'gamma': [g.tolist() for g in self.gamma],
                'beta': [b.tolist() for b in self.beta],
                'running_mean': [rm.tolist() for rm in self.running_mean],
                'running_var': [rv.tolist() for rv in self.running_var]
            }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SimpleNN':
        """Load model from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        arch = model_data['architecture']
        model = cls(
            input_size=arch['input_size'],
            hidden_sizes=arch['hidden_sizes'],
            output_size=arch['output_size'],
            learning_rate=arch['learning_rate'],
            activation=Activation(arch['activation']),
            output_activation=Activation(arch['output_activation']),
            optimizer=Optimizer(arch['optimizer']),
            l2_reg=arch['l2_reg'],
            dropout_rate=arch['dropout_rate'],
            batch_norm=arch['batch_norm'],
            clip_value=arch['clip_value']
        )
        
        # Load weights
        for i, layer in enumerate(model_data['weights']):
            model.layers[i]['W'] = np.array(layer['W'])
            model.layers[i]['b'] = np.array(layer['b'])
        
        # Load optimizer state
        if model_data['optimizer_state'] is not None:
            model.t = model_data['optimizer_state']['t']
            for i, m in enumerate(model_data['optimizer_state']['m']):
                model.m[i]['W'] = np.array(m['W'])
                model.m[i]['b'] = np.array(m['b'])
            for i, v in enumerate(model_data['optimizer_state']['v']):
                model.v[i]['W'] = np.array(v['W'])
                model.v[i]['b'] = np.array(v['b'])
        
        # Load batch norm params
        if model.batch_norm and 'batch_norm' in model_data:
            model.gamma = [np.array(g) for g in model_data['batch_norm']['gamma']]
            model.beta = [np.array(b) for b in model_data['batch_norm']['beta']]
            model.running_mean = [np.array(rm) for rm in model_data['batch_norm']['running_mean']]
            model.running_var = [np.array(rv) for rv in model_data['batch_norm']['running_var']]
        
        return model

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train on a single batch and return loss"""
        # Forward pass in training mode (activates dropout and batch norm)
        output = self.forward(X, training=True)
        
        # Compute loss (MSE)
        loss = np.mean((output - y) ** 2) + \
            (self.l2_reg / (2 * X.shape[0])) * sum(np.sum(layer['W']**2) for layer in self.layers)
        
        # Backward pass to compute gradients and update weights
        self.backward(X, y, output)
        
        return loss

    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss with L2 regularization"""
        m = y_true.shape[0]
        if self.output_activation == Activation.LINEAR:
            loss = np.mean((y_true - y_pred)**2)
        elif self.output_activation == Activation.SIGMOID:
            loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))
        
        # Add L2 regularization
        l2_loss = 0
        for layer in self.layers:
            l2_loss += np.sum(layer['W']**2)
        loss += self.l2_reg * l2_loss / (2 * m)
        
        return float(loss)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with training=False, guaranteeing no dropout/batchnorm mutation"""
        return self.forward(X, training=False)

    def get_state_dict(self) -> dict:
        """
        Return a copy of model parameters where each layer is a dict with 'W' and 'b' numpy arrays.
        """
        layers_state = []
        for layer in self.layers:
            layers_state.append({
                'W': layer['W'].copy(),
                'b': layer['b'].copy()
            })
        return {'layers': layers_state}


    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load parameters from state_dict into self.layers, assuming layers are dicts.
        """
        if 'layers' not in state_dict:
            raise ValueError("Invalid state_dict: missing 'layers' key")
        if len(state_dict['layers']) != len(self.layers):
            raise ValueError("Mismatch in number of layers")

        for i, layer_param in enumerate(state_dict['layers']):
            if 'W' not in layer_param or 'b' not in layer_param:
                raise ValueError("Each layer dict must contain 'W' and 'b'")
            self.layers[i]['W'] = layer_param['W'].copy()
            self.layers[i]['b'] = layer_param['b'].copy()

