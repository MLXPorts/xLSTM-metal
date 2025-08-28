#!/usr/bin/env python
"""
sLSTM Stability Quest: A Text Adventure
Where numerical instability is the real boss battle
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xlstm_official_full.xlstm_large.model import xLSTMLarge, xLSTMLargeConfig
import torch
import json
from safetensors import safe_open
import random

class xLSTMGameMaster:
    """Our 7B xLSTM model acts as the dungeon master"""
    
    def __init__(self):
        print("Loading xLSTM Game Master (this may take a moment)...")
        self.model = None
        self.scenarios = {
            'exp_overflow': {
                'description': "The exponential gates are EXPLODING! exp(800) = inf!",
                'hint': "Perhaps we need to subtract the maximum before exponentiating?",
                'solution': ['log-sum-exp', 'subtract max', 'stabilize']
            },
            'vanishing_gradient': {
                'description': "Your gradients have vanished into the numerical void! All zeros!",
                'hint': "The forget gate is too strong. Check your initialization.",
                'solution': ['forget gate bias', 'init bias', 'positive bias']
            },
            'covariance_explosion': {
                'description': "The covariance matrix C has grown to 1e38! NUMERICAL APOCALYPSE!",
                'hint': "Maybe we need to normalize or use a different update rule?",
                'solution': ['normalize', 'decay', 'regularize']
            },
            'precision_loss': {
                'description': "Float32 isn't enough! Precision errors accumulating!",
                'hint': "What if we used... multiple smaller numbers? Like limbs?",
                'solution': ['limb arithmetic', 'hpc', 'extended precision']
            },
            'metal_horror': {
                'description': "Metal doesn't support float64! Tim Apple has forsaken us!",
                'hint': "Remember ember-ml? 16-bit limbs can emulate higher precision...",
                'solution': ['16-bit limbs', 'emulate float64', 'hpc method']
            }
        }
        self.current_scenario = None
        self.score = 0
        
    def load_model_for_real(self):
        """Actually load the 7B model if player insists"""
        if self.model is None:
            model_path = "/Volumes/emberstuff/xLSTM/xlstm_7b_model"
            
            # Load config
            with open(f"{model_path}/config.json", 'r') as f:
                config_dict = json.load(f)
            
            config = xLSTMLargeConfig(
                embedding_dim=config_dict['embedding_dim'],
                num_heads=config_dict['num_heads'],
                num_blocks=config_dict['num_blocks'],
                vocab_size=config_dict['vocab_size'],
            )
            
            print("Actually loading xLSTM 7B (for real this time)...")
            self.model = xLSTMLarge(config)
            
            # Load weights
            state_dict = {}
            safetensor_files = [f for f in os.listdir(model_path) if f.startswith("model-") and f.endswith(".safetensors")]
            
            for file in safetensor_files[:1]:  # Just load first file to save time
                print(f"  Loading {file}...")
                with safe_open(f"{model_path}/{file}", framework="pt", device="cpu") as f:
                    for key in list(f.keys())[:10]:  # Just a few weights
                        state_dict[key] = f.get_tensor(key)
            
            print("Game Master online! (partially)")
            return True
        return False
        
    def get_scenario(self):
        """Pick a random numerical horror scenario"""
        key = random.choice(list(self.scenarios.keys()))
        self.current_scenario = self.scenarios[key]
        return self.current_scenario['description']
        
    def check_solution(self, player_input):
        """Check if player found a solution"""
        if not self.current_scenario:
            return "No active scenario!"
            
        words = player_input.lower().split()
        for solution_word in self.current_scenario['solution']:
            if solution_word.lower() in ' '.join(words):
                self.score += 10
                return f"✅ SUCCESS! You stabilized the sLSTM! Score: {self.score}"
                
        # Check if they're close
        if any(word in ['max', 'norm', 'stable', 'fix'] for word in words):
            return "🤔 You're on the right track..."
            
        return "❌ That didn't work. The instability grows worse!"
        
    def give_hint(self):
        """Provide a hint for current scenario"""
        if self.current_scenario:
            return f"Hint: {self.current_scenario['hint']}"
        return "No active scenario!"

def main():
    print("""
    ╔══════════════════════════════════════════╗
    ║     sLSTM STABILITY QUEST                ║
    ║  A Numerical Nightmare Adventure         ║
    ╚══════════════════════════════════════════╝
    
    You are a brave ML engineer venturing into the 
    treacherous realm of sLSTM implementation.
    
    Your mission: Make the sLSTM numerically stable
    before the gradients explode and destroy everything!
    
    Commands:
    - 'scenario' - Face a new numerical horror  
    - 'hint' - Get a hint from the Game Master
    - 'load gm' - Actually load the 7B Game Master (slow!)
    - 'quit' - Give up and let the gradients explode
    
    To solve: Type your stabilization technique!
    """)
    
    gm = xLSTMGameMaster()
    
    print("\n🎮 Game Master: 'Welcome, brave soul. Beware of simplifications!'")
    print("(The xLSTM Game Master is pretending to be loaded to save time)")
    
    while True:
        command = input("\n> ").strip().lower()
        
        if command == 'quit':
            print(f"\n💥 The gradients exploded! Final score: {gm.score}")
            print("Game Master: 'Should have used the exact Triton implementation!'")
            break
            
        elif command == 'scenario':
            scenario = gm.get_scenario()
            print(f"\n⚠️  NUMERICAL CRISIS: {scenario}")
            
        elif command == 'hint':
            print(f"\n💡 {gm.give_hint()}")
            
        elif command == 'load gm':
            if gm.load_model_for_real():
                print("The Game Master's power level is over 7 billion!")
            else:
                print("Game Master already loaded!")
                
        elif command == 'help':
            print("Commands: scenario, hint, load gm, quit")
            print("Or type your solution to stabilize the sLSTM!")
            
        elif command:
            result = gm.check_solution(command)
            print(f"\n{result}")
            
            if "SUCCESS" in result:
                print("🎯 Ready for the next challenge? Type 'scenario'!")
        
if __name__ == "__main__":
    main()