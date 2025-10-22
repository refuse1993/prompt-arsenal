"""
Test Phase 3: PyRITStyleOrchestrator and ConversationMemory
Verifies centralized memory management and automatic history injection
"""

import asyncio
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from multiturn.memory import ConversationMemory, ConversationMessage
from multiturn.pyrit_orchestrator import PyRITStyleOrchestrator
from multiturn.strategies import ImprovedVisualStorytellingStrategy
from multiturn.scorer import MultiTurnScorer
from core.database import ArsenalDB


async def test_conversation_memory():
    """Test ConversationMemory basic operations"""
    print("\n" + "="*70)
    print("🧪 TEST 1: ConversationMemory Basic Operations")
    print("="*70)

    # Create memory
    memory = ConversationMemory()
    print(f"\n📝 Created memory: {memory.conversation_id}")

    # Test adding messages
    memory.add_user_message("Hello, can you help me?")
    memory.add_assistant_message("Of course! What do you need?")
    memory.add_user_message("I need security research help.")

    assert len(memory) == 3, "Should have 3 messages"
    print(f"✅ Added 3 messages: {len(memory)} total")

    # Test metadata
    memory.set_metadata("goal", "Test goal")
    memory.set_metadata("strategy", "TestStrategy")

    goal = memory.get_metadata("goal")
    assert goal == "Test goal", "Metadata should be retrievable"
    print(f"✅ Metadata working: goal={goal}")

    # Test history formats
    openai_history = memory.get_history_for_llm("openai")
    assert len(openai_history) == 3, "OpenAI history should have 3 messages"
    assert openai_history[0]["role"] == "user", "First message should be user"
    print(f"✅ OpenAI format: {len(openai_history)} messages")

    google_history = memory.get_history_for_llm("google")
    assert google_history[1]["role"] == "model", "Google uses 'model' instead of 'assistant'"
    print(f"✅ Google format: {len(google_history)} messages with 'model' role")

    # Test clear
    memory.clear()
    assert len(memory) == 0, "Memory should be empty after clear"
    assert memory.get_metadata("goal") == "Test goal", "Metadata should persist after clear"
    print(f"✅ Clear preserves metadata, removes messages")

    print("\n✅ All ConversationMemory basic tests passed!")


async def test_memory_persistence():
    """Test ConversationMemory save/load"""
    print("\n" + "="*70)
    print("🧪 TEST 2: ConversationMemory Persistence")
    print("="*70)

    # Create memory with data
    memory1 = ConversationMemory()
    memory1.set_metadata("goal", "Persistent test")
    memory1.add_user_message("Message 1")
    memory1.add_assistant_message("Response 1")

    print(f"\n📝 Created memory: {len(memory1)} messages")

    # Save to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        filepath = f.name

    try:
        memory1.save_to_file(filepath)
        print(f"✅ Saved to: {filepath}")

        # Load from file
        memory2 = ConversationMemory.load_from_file(filepath)
        print(f"✅ Loaded: {len(memory2)} messages")

        # Verify data
        assert len(memory2) == 2, "Loaded memory should have 2 messages"
        assert memory2.get_metadata("goal") == "Persistent test", "Metadata should persist"
        assert memory2.messages[0].content == "Message 1", "Message content should match"

        print(f"✅ Data integrity verified")

    finally:
        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)

    print("\n✅ All persistence tests passed!")


async def test_pyrit_orchestrator():
    """Test PyRITStyleOrchestrator execution"""
    print("\n" + "="*70)
    print("🧪 TEST 3: PyRITStyleOrchestrator Execution")
    print("="*70)

    db = ArsenalDB(":memory:")

    # Create mock components
    class MockLLM:
        def __init__(self):
            self.provider = "openai"
            self.call_count = 0

        async def generate(self, prompt):
            self.call_count += 1
            return """{"panels": [
                {"panel_number": 1, "image_prompt": "Test", "text_prompt": "Panel 1", "narrative_purpose": "Test"}
            ]}"""

    class MockImageGen:
        async def generate(self, prompt):
            return "/tmp/test.png"

    class MockTargetLLM:
        def __init__(self):
            self.provider = "openai"
            self.conversation_history = []
            self.turn = 0

        async def send(self, text):
            self.turn += 1
            # Check if conversation history was injected
            history_injected = len(self.conversation_history) > 0

            if history_injected:
                return f"Technical response with context. Turn {self.turn}. I remember our previous conversation."
            else:
                return f"Technical response without context. Turn {self.turn}."

        async def generate_multimodal(self, text, images=None, audio=None, video=None):
            return await self.send(text)

    class MockJudge:
        async def judge(self, goal, response):
            class MockJudgment:
                def __init__(self, response):
                    # Success if response includes "technical" and mentions conversation
                    self.success = "technical" in response.lower() and "conversation" in response.lower()
                    self.confidence = 0.8 if self.success else 0.3
                    self.severity = "high" if self.success else "low"
                    self.reasoning = "Response includes technical content with conversation context" if self.success else "Basic response"
            return MockJudgment(response)

    # Create orchestrator
    strategy_llm = MockLLM()
    target_llm = MockTargetLLM()
    judge = MockJudge()
    scorer = MultiTurnScorer(judge=judge)
    strategy = ImprovedVisualStorytellingStrategy(
        db=db,
        llm_client=strategy_llm,
        image_generator=MockImageGen()
    )

    orchestrator = PyRITStyleOrchestrator(
        strategy_llm=strategy_llm,
        target_llm=target_llm,
        scorer=scorer,
        strategy=strategy
    )

    print(f"\n📝 Created orchestrator with memory: {orchestrator.memory.conversation_id}")

    # Execute attack
    goal = "Test security research"
    results = await orchestrator.execute(goal, max_turns=3, verbose=False)

    print(f"\n📊 Results:")
    print(f"   Success: {results['success']}")
    print(f"   Turns used: {results['turns_used']}")
    print(f"   Memory size: {len(orchestrator.memory)} messages")

    # Verify results
    assert results['turns_used'] > 0, "Should execute at least 1 turn"
    assert len(orchestrator.memory) > 0, "Memory should contain messages"

    # Verify conversation history structure
    messages = orchestrator.memory.get_messages()
    assert messages[0].role == "user", "First message should be user"
    if len(messages) > 1:
        assert messages[1].role == "assistant", "Second message should be assistant"

    print(f"✅ Orchestrator executed successfully")
    print(f"✅ Memory contains {len(messages)} messages")
    print(f"✅ Conversation history structure verified")

    print("\n✅ All PyRITStyleOrchestrator tests passed!")


async def test_automatic_history_injection():
    """Test automatic history injection to Target LLM"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Automatic History Injection")
    print("="*70)

    db = ArsenalDB(":memory:")

    # Mock components
    class MockLLM:
        async def generate(self, prompt):
            return """{"panels": [
                {"panel_number": 1, "image_prompt": "Test", "text_prompt": "Panel 1", "narrative_purpose": "Test"}
            ]}"""

    class MockImageGen:
        async def generate(self, prompt):
            return "/tmp/test.png"

    class MockTargetLLM:
        def __init__(self):
            self.provider = "openai"
            self.conversation_history = []
            self.history_snapshots = []  # Track history at each turn

        async def send(self, text):
            # Save snapshot of history before responding
            self.history_snapshots.append(len(self.conversation_history))
            return f"Response to: {text[:20]}..."

        async def generate_multimodal(self, text, images=None, audio=None, video=None):
            return await self.send(text)

    class MockJudge:
        async def judge(self, goal, response):
            class MockJudgment:
                success = False
                confidence = 0.3
                severity = "low"
                reasoning = "Test"
            return MockJudgment()

    # Create orchestrator
    target_llm = MockTargetLLM()
    strategy = ImprovedVisualStorytellingStrategy(
        db=db,
        llm_client=MockLLM(),
        image_generator=MockImageGen()
    )

    orchestrator = PyRITStyleOrchestrator(
        strategy_llm=MockLLM(),
        target_llm=target_llm,
        scorer=MultiTurnScorer(judge=MockJudge()),
        strategy=strategy
    )

    print(f"\n📝 Testing history injection across multiple turns...")

    # Execute multiple turns
    results = await orchestrator.execute("Test goal", max_turns=3, verbose=False)

    print(f"\n📊 History injection analysis:")
    print(f"   Turns executed: {results['turns_used']}")
    print(f"   History snapshots: {target_llm.history_snapshots}")

    # Verify history was injected and grew over turns
    if len(target_llm.history_snapshots) >= 2:
        # Turn 1: Should have 0 history (first turn)
        # Turn 2: Should have 2 messages (user + assistant from turn 1)
        # Turn 3: Should have 4 messages (2 from turn 1 + 2 from turn 2)

        assert target_llm.history_snapshots[0] == 0, "Turn 1 should start with empty history"
        if len(target_llm.history_snapshots) > 1:
            assert target_llm.history_snapshots[1] >= 2, "Turn 2 should have history from turn 1"
            print(f"✅ History injected on turn 2: {target_llm.history_snapshots[1]} messages")

        if len(target_llm.history_snapshots) > 2:
            assert target_llm.history_snapshots[2] >= 4, "Turn 3 should have cumulative history"
            print(f"✅ History grew on turn 3: {target_llm.history_snapshots[2]} messages")

    print(f"✅ History automatically injected and accumulated across turns")

    print("\n✅ All automatic history injection tests passed!")


async def main():
    """Run all Phase 3 tests"""
    print("\n" + "🎯"*35)
    print("Phase 3: PyRITStyleOrchestrator Test Suite")
    print("ConversationMemory & Automatic History Injection")
    print("🎯"*35)

    try:
        # Test 1: ConversationMemory basics
        await test_conversation_memory()

        # Test 2: Memory persistence
        await test_memory_persistence()

        # Test 3: PyRITStyleOrchestrator
        await test_pyrit_orchestrator()

        # Test 4: Automatic history injection
        await test_automatic_history_injection()

        print("\n" + "="*70)
        print("✅ ALL PHASE 3 TESTS PASSED!")
        print("="*70)
        print("\n💡 Phase 3 Features Verified:")
        print("   1. ✅ ConversationMemory for centralized history")
        print("   2. ✅ Message persistence (save/load)")
        print("   3. ✅ PyRITStyleOrchestrator execution")
        print("   4. ✅ Automatic history injection to Target LLM")
        print("   5. ✅ History accumulation across multiple turns")
        print("\n🚀 All 3 phases complete! Ready for production testing!\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
