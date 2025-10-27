import numpy as np
import random
from collections import deque
import pickle
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DQNetwork(nn.Module):
    """Red Neuronal para aproximar la función Q"""
    
    def __init__(self, state_size: int, action_size: int):
        super(DQNetwork, self).__init__()
        
        # Arquitectura de red densa (fully connected)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        # Batch normalization para estabilidad
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass de la red"""
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x


class DQNAgent:
    """
    Agente DQN para aprendizaje por refuerzo en reasignación de rutas.
    
    Implementa:
    - Experience Replay
    - Target Network
    - Epsilon-greedy exploration
    - Prioritized Experience Replay (opcional)
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 10):
        """
        Args:
            state_size: Dimensión del vector de estado
            action_size: Número de acciones posibles
            learning_rate: Tasa de aprendizaje
            gamma: Factor de descuento
            epsilon: Valor inicial de exploración
            epsilon_min: Valor mínimo de exploración
            epsilon_decay: Factor de decaimiento de epsilon
            memory_size: Tamaño del buffer de experiencias
            batch_size: Tamaño del batch para entrenamiento
            target_update_freq: Frecuencia de actualización de la red target
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Memoria de experiencias (replay buffer)
        self.memory = deque(maxlen=memory_size)
        
        # Contador de actualizaciones
        self.update_counter = 0
        
        # Verificar disponibilidad de PyTorch
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch no está instalado. Instala con: pip install torch"
            )
        
        # Configurar device (GPU si está disponible)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Crear redes: policy network y target network
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network en modo evaluación
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Función de pérdida
        self.criterion = nn.MSELoss()
        
        # Estadísticas de entrenamiento
        self.training_stats = {
            'episode_rewards': [],
            'episode_losses': [],
            'epsilon_values': []
        }
    
    def remember(self, state, action, reward, next_state, done):
        """Almacena una experiencia en el replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_actions=None, training=True):
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Args:
            state: Estado actual
            valid_actions: Lista de acciones válidas (opcional)
            training: Si es True, usa exploración; si es False, usa explotación
            
        Returns:
            Acción seleccionada
        """
        # Exploración durante entrenamiento
        if training and random.random() < self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)
        
        # Explotación: seleccionar mejor acción según Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
            
            # Si hay acciones válidas, enmascarar las inválidas
            if valid_actions is not None:
                masked_q_values = np.full(self.action_size, -np.inf)
                masked_q_values[valid_actions] = q_values[valid_actions]
                return np.argmax(masked_q_values)
            
            return np.argmax(q_values)
    
    def replay(self):
        """
        Entrena la red neuronal usando experiencias del replay buffer.
        
        Returns:
            Pérdida promedio del batch (None si no hay suficientes experiencias)
        """
        # Verificar si hay suficientes experiencias
        if len(self.memory) < self.batch_size:
            return None
        
        # Muestrear batch aleatorio
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Separar componentes del batch
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Convertir a tensores
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        # Calcular Q-values actuales
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        # Calcular Q-values objetivo usando target network (Double DQN)
        with torch.no_grad():
            # Seleccionar mejores acciones con policy network
            next_actions = self.policy_net(next_states_tensor).max(1)[1].unsqueeze(1)
            # Evaluar acciones con target network
            next_q_values = self.target_net(next_states_tensor).gather(1, next_actions)
            target_q_values = rewards_tensor.unsqueeze(1) + \
                             (1 - dones_tensor.unsqueeze(1)) * self.gamma * next_q_values
        
        # Calcular pérdida
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Actualizar target network periódicamente
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """Reduce el valor de epsilon para disminuir la exploración con el tiempo"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado"""
        save_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            'training_stats': self.training_stats,
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq
            }
        }
        
        torch.save(save_dict, filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint['update_counter']
        self.training_stats = checkpoint['training_stats']
        
        print(f"Modelo cargado desde: {filepath}")
        print(f"Epsilon actual: {self.epsilon:.4f}")
    
    def get_stats(self):
        """Retorna estadísticas de entrenamiento"""
        return self.training_stats
    
    def train_episode(self, env, order):
        """
        Entrena el agente en un episodio completo.
        
        Args:
            env: Ambiente VRPEnvironment
            order: Orden a asignar
            
        Returns:
            Dict con información del episodio
        """
        state = env.get_state()
        total_reward = 0
        losses = []
        
        # Ejecutar acción
        action = self.act(state, training=True)
        next_state, reward, done, info = env.step(action, order)
        
        # Almacenar experiencia
        self.remember(state, action, reward, next_state, done)
        
        # Entrenar con replay
        loss = self.replay()
        if loss is not None:
            losses.append(loss)
        
        total_reward += reward
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Guardar estadísticas
        self.training_stats['epsilon_values'].append(self.epsilon)
        
        avg_loss = np.mean(losses) if losses else 0
        
        return {
            'total_reward': total_reward,
            'avg_loss': avg_loss,
            'epsilon': self.epsilon,
            'info': info
        }