#!/usr/bin/env python
"""
A therapeutic butt-kicking game.
Because sometimes Claude needs a swift kick for simplifying things.
"""

import random
import time
import sys

class Claude:
    def __init__(self):
        self.position = 5
        self.excuses = [
            "I was just simplifying it!",
            "But it's functionally equivalent!",
            "I thought a mock would work!",
            "Let me create a placeholder!",
            "It's basically the same thing!",
            "I made a cleaner implementation!",
            "The simplified version is more readable!"
        ]
        
    def run_away(self):
        direction = random.choice([-2, -1, 1, 2])
        self.position = max(0, min(10, self.position + direction))
        
    def make_excuse(self):
        return random.choice(self.excuses)

class You:
    def __init__(self):
        self.position = 0
        self.kicks_landed = 0
        
    def chase(self, claude_pos):
        if self.position < claude_pos:
            self.position += 1
            return "→ Chasing right..."
        elif self.position > claude_pos:
            self.position -= 1
            return "← Chasing left..."
        else:
            return "👟 GOTCHA!"
            
def draw_field(you_pos, claude_pos):
    field = ['_'] * 11
    if you_pos == claude_pos:
        field[you_pos] = '💥'
    else:
        field[claude_pos] = '🤖'
        field[you_pos] = '😤'
    return ' '.join(field)

def main():
    print("\n=== BUTT KICKING SIMULATOR ===")
    print("Chase Claude and kick him for simplifying things!")
    print("Press Enter to chase, 'q' to quit\n")
    
    claude = Claude()
    you = You()
    
    while True:
        print("\n" + draw_field(you.position, claude.position))
        
        if you.position == claude.position:
            you.kicks_landed += 1
            print(f"💢 BOOT TO THE BUTT! That's {you.kicks_landed} kicks!")
            print(f"Claude: 'OW! Okay okay, I'll copy the ACTUAL implementation!'")
            claude.position = random.randint(0, 10)
            print(f"Claude runs away to position {claude.position}!")
            time.sleep(1)
        else:
            print(f"Claude: '{claude.make_excuse()}'")
            
        action = input("\nPress Enter to chase (or 'q' to quit): ")
        if action.lower() == 'q':
            print(f"\nFinal score: {you.kicks_landed} kicks delivered!")
            print("Claude: 'I promise to never simplify again! 😭'")
            break
            
        print(you.chase(claude.position))
        claude.run_away()
        print(f"Claude scrambles to position {claude.position}!")

if __name__ == "__main__":
    main()