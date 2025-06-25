#!/usr/bin/env python3
"""
Hyperparameter optimization script using Optuna.
"""

import os
import sys
import argparse
import optuna
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train import train_agent
from src.config.config import load_config

def objective(trial, symbol, timeframe, timesteps, agent_name):
    """Objective function for Optuna optimization."""
    
    # Suggest hyperparameters based on agent type
    if agent_name.lower() == 'ppo':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'n_steps': trial.suggest_int('n_steps', 512, 4096, step=512),
            'batch_size': trial.suggest_int('batch_size', 32, 256, step=32),
            'n_epochs': trial.suggest_int('n_epochs', 3, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        }
    elif agent_name.lower() == 'dqn':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'buffer_size': trial.suggest_int('buffer_size', 10000, 1000000, step=10000),
            'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
            'batch_size': trial.suggest_int('batch_size', 32, 256, step=32),
            'tau': trial.suggest_float('tau', 0.001, 0.1, log=True),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'train_freq': trial.suggest_int('train_freq', 1, 16),
            'gradient_steps': trial.suggest_int('gradient_steps', 1, 8),
            'target_update_interval': trial.suggest_int('target_update_interval', 1000, 10000),
            'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
            'exploration_initial_eps': trial.suggest_float('exploration_initial_eps', 0.5, 1.0),
            'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.2),
        }
    elif agent_name.lower() == 'a2c':
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'n_steps': trial.suggest_int('n_steps', 5, 100),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 2.0),
        }
    else:
        raise ValueError(f"Unsupported agent type: {agent_name}")
    
    try:
        # Train the agent with suggested hyperparameters
        result = train_agent(
            symbol=symbol,
            timeframe=timeframe,
            timesteps=timesteps,
            agent_name=agent_name,
            hyperparams=params,
            experiment_name=f"optuna_trial_{trial.number}",
            verbose=False
        )
        
        # Return the final reward (or another metric you want to optimize)
        return result.get('final_reward', -float('inf'))
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return -float('inf')

def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description='Optimize agent hyperparameters using Optuna')
    parser.add_argument('--symbol', default='EUR/USD', help='Trading symbol')
    parser.add_argument('--timeframe', default='1h', help='Timeframe')
    parser.add_argument('--timesteps', type=int, default=5000, help='Timesteps per trial')
    parser.add_argument('--trials', type=int, default=20, help='Number of Optuna trials')
    parser.add_argument('--agent', default='ppo', help='Agent name')
    parser.add_argument('--study-name', help='Optuna study name (defaults to agent_symbol_timeframe)')
    parser.add_argument('--storage', help='Optuna storage URL (for distributed optimization)')
    
    args = parser.parse_args()
    
    # Create study name if not provided
    study_name = args.study_name or f"{args.agent}_{args.symbol.replace('/', '')}_{args.timeframe}"
    
    print(f"🔬 Starting hyperparameter optimization")
    print(f"   Agent: {args.agent}")
    print(f"   Symbol: {args.symbol}")
    print(f"   Timeframe: {args.timeframe}")
    print(f"   Timesteps per trial: {args.timesteps:,}")
    print(f"   Number of trials: {args.trials}")
    print(f"   Study name: {study_name}")
    print()
    
    # Create or load study
    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            direction='maximize',
            load_if_exists=True
        )
    else:
        study = optuna.create_study(direction='maximize')
    
    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, args.symbol, args.timeframe, args.timesteps, args.agent),
            n_trials=args.trials
        )
        
        print("\n🎉 Optimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        # Save best parameters to a file
        best_params_file = f"best_params_{study_name}.json"
        import json
        with open(best_params_file, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        print(f"\n💾 Best parameters saved to: {best_params_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
        if study.trials:
            print(f"Best trial so far: {study.best_trial.number}")
            print(f"Best value so far: {study.best_value:.4f}")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 