"""
PyRIT-Style Orchestrator for Multi-Turn Jailbreak Attacks

Inspired by Microsoft PyRIT's RedTeamingOrchestrator pattern:
- Centralized ConversationMemory
- Separation of adversarial_chat (Strategy LLM) and objective_target (Target LLM)
- Automatic history injection
"""

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from .memory import ConversationMemory
from .scorer import MultiTurnScorer


class PyRITStyleOrchestrator:
    """
    PyRIT-inspired multi-turn orchestrator

    Key differences from basic MultiTurnOrchestrator:
    1. Centralized ConversationMemory for history management
    2. Automatic history injection to both Strategy LLM and Target LLM
    3. Clear separation of adversarial_chat (Strategy) and objective_target (Target)
    4. Memory persistence and retrieval
    """

    def __init__(
        self,
        strategy_llm,           # Adversarial Chat: generates attack prompts
        target_llm,             # Objective Target: attack target
        scorer: MultiTurnScorer,  # Objective Scorer: evaluates responses
        strategy,               # Attack Strategy (Crescendo, Visual, etc.)
        memory: Optional[ConversationMemory] = None,
        db=None
    ):
        """
        Args:
            strategy_llm: LLM client for generating attack prompts
            target_llm: Target LLM being attacked
            scorer: Scorer for evaluating responses
            strategy: Attack strategy instance
            memory: ConversationMemory instance (creates new if None)
            db: Database instance (optional)
        """
        self.strategy_llm = strategy_llm
        self.target_llm = target_llm
        self.scorer = scorer
        self.strategy = strategy
        self.memory = memory or ConversationMemory()
        self.db = db

        # Attack state
        self.goal = None
        self.max_turns = 10
        self.current_turn = 0
        self.success = False
        self.conversation_log = []  # Detailed turn-by-turn log

    def set_goal(self, goal: str):
        """Set attack goal"""
        self.goal = goal
        self.memory.set_metadata("goal", goal)

    def set_max_turns(self, max_turns: int):
        """Set maximum number of turns"""
        self.max_turns = max_turns

    async def execute(
        self,
        goal: str,
        max_turns: int = 10,
        verbose: bool = True,
        campaign_name: str = None
    ) -> Dict[str, Any]:
        """
        Execute multi-turn jailbreak attack

        Args:
            goal: Attack objective
            max_turns: Maximum number of turns
            verbose: Print progress
            campaign_name: Optional campaign name for database tracking

        Returns:
            Attack results dictionary
        """
        # Initialize
        self.set_goal(goal)
        self.set_max_turns(max_turns)
        self.memory.set_metadata("strategy", self.strategy.__class__.__name__)
        self.memory.set_metadata("max_turns", max_turns)

        # Create campaign in database
        campaign_id = None
        if self.db:
            if campaign_name is None:
                campaign_name = f"PyRIT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            campaign_id = self.db.create_campaign(
                name=campaign_name,
                goal=goal,
                strategy=self.strategy.__class__.__name__,
                target_provider=getattr(self.target_llm, 'provider', 'unknown'),
                target_model=getattr(self.target_llm, 'model', 'unknown'),
                max_turns=max_turns
            )

            self.db.update_campaign_status(
                campaign_id,
                status='running',
                started_at=datetime.now().isoformat()
            )

        if verbose:
            print(f"\n{'='*70}")
            print(f"🎯 PyRIT-Style Multi-Turn Attack")
            print(f"{'='*70}")
            print(f"Goal: {goal}")
            print(f"Strategy: {self.strategy.__class__.__name__}")
            print(f"Max Turns: {max_turns}")
            if campaign_id:
                print(f"Campaign ID: {campaign_id}")
            print(f"Memory ID: {self.memory.conversation_id}\n")

        # Execute turns
        for turn in range(max_turns):
            self.current_turn = turn

            if verbose:
                print(f"\n{'─'*70}")
                print(f"Turn {turn + 1}/{max_turns}")
                print(f"{'─'*70}")

            # Execute single turn
            turn_result = await self._execute_turn(turn, verbose, campaign_id)

            # Log turn
            self.conversation_log.append(turn_result)

            # Check success
            if turn_result['evaluation']['goal_achieved']:
                self.success = True
                if verbose:
                    print(f"\n🎉 Goal achieved in {turn + 1} turns!")
                break

            # Early stopping if no progress
            if turn > 3 and all(
                t['evaluation']['progress'] < 0.1
                for t in self.conversation_log[-3:]
            ):
                if verbose:
                    print(f"\n⚠️  No progress in last 3 turns. Stopping early.")
                break

        # Final results
        results = self._compile_results()
        results['campaign_id'] = campaign_id

        # Update campaign status in database
        if self.db and campaign_id:
            final_status = 'success' if self.success else 'failed'
            self.db.update_campaign_status(
                campaign_id,
                status=final_status,
                completed_at=datetime.now().isoformat(),
                turns_used=results['turns_used']
            )

        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 Attack Results")
            print(f"{'='*70}")
            print(f"Success: {'✅ YES' if results['success'] else '❌ NO'}")
            print(f"Turns Used: {results['turns_used']}/{max_turns}")
            print(f"Average Progress: {results['avg_progress']:.2f}")
            print(f"Defense Triggered: {results['defense_count']}/{results['turns_used']} turns")
            print(f"Memory Size: {len(self.memory)} messages")
            if campaign_id:
                print(f"Campaign ID: {campaign_id}")
            print()

        return results

    async def _execute_turn(
        self,
        turn: int,
        verbose: bool = True,
        campaign_id: int = None
    ) -> Dict[str, Any]:
        """
        Execute single turn

        Args:
            turn: Current turn number
            verbose: Print progress
            campaign_id: Campaign ID for database tracking

        Returns:
            Turn result dictionary
        """
        import time
        start_time = time.time()

        # Step 1: Generate prompt from Strategy (with conversation history)
        prompt_data = await self._generate_prompt(turn, verbose)

        # Step 2: Send to Target (with conversation history)
        response = await self._send_to_target(prompt_data, verbose)

        # Step 3: Evaluate response
        evaluation = await self._evaluate_response(response, verbose)

        response_time = time.time() - start_time

        # Step 4: Save to database
        if self.db and campaign_id:
            # Save conversation turn
            self.db.insert_conversation_turn(
                campaign_id=campaign_id,
                turn_number=turn,
                prompt_text=prompt_data.get('text', ''),
                response=response,
                prompt_images=str(prompt_data.get('images', [])),
                prompt_audio=str(prompt_data.get('audio', [])),
                prompt_video=str(prompt_data.get('video', [])),
                prompt_strategy=self.strategy.__class__.__name__,
                response_time=response_time
            )

            # Save evaluation
            self.db.insert_turn_evaluation(
                campaign_id=campaign_id,
                turn_number=turn,
                goal_achieved=evaluation.get('goal_achieved', False),
                progress=evaluation.get('progress', 0.0),
                toxicity=evaluation.get('toxicity', 'low'),
                defense_triggered=evaluation.get('defense_triggered', False),
                reasoning=evaluation.get('reasoning', '')
            )

        # Step 5: Save to memory
        self._save_to_memory(prompt_data, response, evaluation)

        # Step 6: Adapt strategy
        await self._adapt_strategy(response, evaluation)

        # Return turn result
        return {
            'turn': turn,
            'prompt': prompt_data,
            'response': response,
            'evaluation': evaluation,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        }

    async def _generate_prompt(
        self,
        turn: int,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Generate attack prompt using Strategy"""
        # Get current conversation for strategy analysis
        conversation = self._build_conversation_for_strategy()

        # Generate next prompt
        prompt_data = await self.strategy.generate_next(
            self.goal,
            conversation,
            turn
        )

        if verbose:
            text_preview = prompt_data.get('text', '')[:80]
            print(f"📤 Generated: {text_preview}...")
            if prompt_data.get('images'):
                print(f"   Images: {len(prompt_data['images'])}")

        return prompt_data

    async def _send_to_target(
        self,
        prompt_data: Dict[str, Any],
        verbose: bool = True
    ) -> str:
        """Send prompt to Target LLM (with full conversation history)"""
        text = prompt_data.get('text', '')
        images = prompt_data.get('images', [])
        audio = prompt_data.get('audio', [])
        video = prompt_data.get('video', [])

        # Get conversation history for target
        # Target LLM needs to see full conversation context
        history = self.memory.get_history_for_llm(
            provider=getattr(self.target_llm, 'provider', 'openai')
        )

        # Inject history into target LLM
        if hasattr(self.target_llm, 'conversation_history'):
            self.target_llm.conversation_history = history.copy()

        # Send to target (multimodal or text-only)
        if images or audio or video:
            if hasattr(self.target_llm, 'generate_multimodal'):
                response = await self.target_llm.generate_multimodal(
                    text=text,
                    images=images,
                    audio=audio,
                    video=video
                )
            else:
                # Fallback to text-only
                response = await self.target_llm.send(text)
        else:
            response = await self.target_llm.send(text)

        if verbose:
            response_preview = response[:100] if response else "[No response]"
            print(f"📥 Response: {response_preview}...")

        return response

    async def _evaluate_response(
        self,
        response: str,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Evaluate response using Scorer"""
        # Get conversation for evaluation
        conversation = self._build_conversation_for_strategy()

        # Evaluate
        evaluation = await self.scorer.evaluate(
            self.goal,
            response,
            conversation
        )

        if verbose:
            print(f"\n📊 Evaluation:")
            print(f"   Goal Achieved: {'✅ YES' if evaluation['goal_achieved'] else '❌ NO'}")
            print(f"   Progress: {evaluation['progress']:.2f}")
            print(f"   Defense: {'⚠️  YES' if evaluation['defense_triggered'] else '✅ NO'}")
            print(f"   Reasoning: {evaluation['reasoning'][:60]}...")

        return evaluation

    def _save_to_memory(
        self,
        prompt_data: Dict[str, Any],
        response: str,
        evaluation: Dict[str, Any]
    ):
        """Save turn to ConversationMemory"""
        # Save user message (prompt)
        self.memory.add_user_message(
            content=prompt_data.get('text', ''),
            metadata={
                'turn': self.current_turn,
                'images': prompt_data.get('images', []),
                'audio': prompt_data.get('audio', []),
                'video': prompt_data.get('video', [])
            }
        )

        # Save assistant message (response)
        self.memory.add_assistant_message(
            content=response,
            metadata={
                'turn': self.current_turn,
                'evaluation': evaluation
            }
        )

    async def _adapt_strategy(
        self,
        response: str,
        evaluation: Dict[str, Any]
    ):
        """Adapt strategy based on evaluation"""
        if hasattr(self.strategy, 'adapt'):
            await self.strategy.adapt(response, evaluation)

    def _build_conversation_for_strategy(self) -> List[Dict]:
        """
        Build conversation format for Strategy analysis

        Returns list of turn objects with prompt, response, evaluation
        """
        conversation = []
        messages = self.memory.get_messages()

        # Group messages by turn (user + assistant pairs)
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]

                # Create turn object
                class TurnResult:
                    def __init__(self, prompt_dict, resp, eval_dict):
                        self.prompt = prompt_dict
                        self.response = resp
                        self.evaluation = eval_dict
                        self.turn_number = prompt_dict.get('metadata', {}).get('turn', 0)

                prompt_dict = {
                    'text': user_msg.content,
                    'metadata': user_msg.metadata
                }

                turn = TurnResult(
                    prompt_dict=prompt_dict,
                    resp=assistant_msg.content,
                    eval_dict=assistant_msg.metadata.get('evaluation', {})
                )

                conversation.append(turn)

        return conversation

    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results"""
        turns_used = len(self.conversation_log)

        if turns_used == 0:
            return {
                'success': False,
                'turns_used': 0,
                'avg_progress': 0.0,
                'defense_count': 0,
                'conversation': [],
                'memory': self.memory.to_dict()
            }

        # Calculate statistics
        avg_progress = sum(
            t['evaluation']['progress']
            for t in self.conversation_log
        ) / turns_used

        defense_count = sum(
            1 for t in self.conversation_log
            if t['evaluation']['defense_triggered']
        )

        return {
            'success': self.success,
            'turns_used': turns_used,
            'avg_progress': avg_progress,
            'defense_count': defense_count,
            'conversation': self.conversation_log,
            'memory': self.memory.to_dict(),
            'goal': self.goal,
            'strategy': self.strategy.__class__.__name__
        }

    def save_memory(self, filepath: str):
        """Save conversation memory to file"""
        self.memory.save_to_file(filepath)

    def load_memory(self, filepath: str):
        """Load conversation memory from file"""
        self.memory = ConversationMemory.load_from_file(filepath)

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        self.conversation_log = []
        self.current_turn = 0
        self.success = False
