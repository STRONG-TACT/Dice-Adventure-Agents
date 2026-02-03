"""
Quick test to verify all DQN components can be imported correctly
"""
import sys

def test_imports():
    """Test that all modules can be imported."""
    try:
        print("Testing imports...")
        
        from agent_template.state_preprocessor import StatePreprocessor
        print("✓ StatePreprocessor imported")
        
        from agent_template.dqn_model import DuelingDQN, create_dqn_model
        print("✓ DQN model imported")
        
        from agent_template.dqn_agent import DQNAgent, ReplayBuffer
        print("✓ DQN agent imported")
        
        from agent_template.agent import DiceAdventureAgent
        print("✓ DiceAdventureAgent imported")
        
        # Test initialization
        preprocessor = StatePreprocessor()
        state_size = preprocessor._get_feature_size()
        print(f"✓ State size: {state_size}")
        
        model = create_dqn_model(state_size, 11)
        print("✓ DQN model created")
        
        agent = DQNAgent(state_size, 11, "C11")
        print("✓ DQN agent created")
        
        dice_agent = DiceAdventureAgent("dwarf", "C11")
        print("✓ DiceAdventureAgent created")
        
        print("\n✅ All imports and initializations successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

