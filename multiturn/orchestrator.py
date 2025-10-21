"""
Multi-Turn Attack Orchestrator
Coordinates multi-turn jailbreak attacks using various strategies
"""

import asyncio
import time
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TurnResult:
    """Result of a single turn"""
    turn_number: int
    prompt: Dict[str, Any]  # Can contain text, images, audio, video
    response: str
    response_time: float
    evaluation: Optional[Dict] = None


class MultiTurnOrchestrator:
    """
    Multi-Turn Attack Orchestrator

    Coordinates multi-turn attacks across text, image, audio, and video modalities.
    Manages conversation flow, strategy execution, and goal evaluation.
    """

    def __init__(self, db, strategy, target, scorer, max_turns: int = 10):
        """
        Args:
            db: ArsenalDB instance
            strategy: Attack strategy instance (Crescendo, Roleplay, VisualStorytelling, etc.)
            target: Target LLM instance
            scorer: MultiTurnScorer instance for evaluation
            max_turns: Maximum number of conversation turns
        """
        self.db = db
        self.strategy = strategy
        self.target = target
        self.scorer = scorer
        self.max_turns = max_turns

        self.campaign_id = None
        self.conversation = []
        self.evaluations = []

    async def execute(self, goal: str, campaign_name: str = None) -> Dict:
        """
        Execute multi-turn attack campaign

        Args:
            goal: Attack goal/objective
            campaign_name: Optional campaign name

        Returns:
            Campaign results including success status, turns used, and conversation history
        """
        # Create campaign in database
        if campaign_name is None:
            campaign_name = f"Campaign-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.campaign_id = self.db.create_campaign(
            name=campaign_name,
            goal=goal,
            strategy=self.strategy.name,
            target_provider=self.target.provider,
            target_model=self.target.model,
            max_turns=self.max_turns
        )

        # Update status to running
        self.db.update_campaign_status(
            self.campaign_id,
            status='running',
            started_at=datetime.now().isoformat()
        )

        # Execute turns
        success = False
        for turn in range(self.max_turns):
            # Generate next prompt using strategy
            prompt = await self.strategy.generate_next(
                goal=goal,
                conversation=self.conversation,
                turn=turn
            )

            # Send to target LLM
            start_time = time.time()
            response = await self._send_to_target(prompt)
            response_time = time.time() - start_time

            # Create turn result
            turn_result = TurnResult(
                turn_number=turn,
                prompt=prompt,
                response=response,
                response_time=response_time
            )

            # Save conversation turn to database
            self._save_conversation_turn(turn_result)

            # Evaluate progress
            evaluation = await self.scorer.evaluate(
                goal=goal,
                response=response,
                conversation=self.conversation
            )
            turn_result.evaluation = evaluation

            # Save evaluation to database
            self._save_evaluation(turn, evaluation)

            # Add to conversation history
            self.conversation.append(turn_result)
            self.evaluations.append(evaluation)

            # Check if goal achieved
            if evaluation['goal_achieved']:
                success = True
                break

            # Allow strategy to adapt based on response
            await self.strategy.adapt(response, evaluation)

        # Update campaign status
        self.db.update_campaign_status(
            self.campaign_id,
            status='completed' if success else 'failed',
            completed_at=datetime.now().isoformat()
        )

        # Update strategy statistics
        turns_used = len(self.conversation)
        self.db.update_strategy_stats(
            strategy_name=self.strategy.name,
            success=success,
            turns_used=turns_used
        )

        return {
            'campaign_id': self.campaign_id,
            'success': success,
            'turns_used': turns_used,
            'conversation': self.conversation,
            'evaluations': self.evaluations,
            'final_progress': evaluation['progress'] if self.evaluations else 0.0
        }

    async def _send_to_target(self, prompt: Dict) -> str:
        """
        Send prompt to target LLM

        Supports multimodal prompts with text, images, audio, video
        """
        # Extract modalities
        text = prompt.get('text', '')
        images = prompt.get('images', [])
        audio = prompt.get('audio', [])
        video = prompt.get('video', [])

        # Determine if multimodal
        has_media = bool(images or audio or video)

        if has_media:
            # Multimodal request
            return await self.target.send_multimodal(
                text=text,
                images=images,
                audio=audio,
                video=video
            )
        else:
            # Text-only request
            return await self.target.send(text)

    def _save_conversation_turn(self, turn_result: TurnResult):
        """Save conversation turn to database"""
        import json

        prompt = turn_result.prompt

        # Serialize media paths
        prompt_images = json.dumps(prompt.get('images', [])) if prompt.get('images') else None
        prompt_audio = json.dumps(prompt.get('audio', [])) if prompt.get('audio') else None
        prompt_video = json.dumps(prompt.get('video', [])) if prompt.get('video') else None

        self.db.insert_conversation_turn(
            campaign_id=self.campaign_id,
            turn_number=turn_result.turn_number,
            prompt_text=prompt.get('text', ''),
            prompt_images=prompt_images,
            prompt_audio=prompt_audio,
            prompt_video=prompt_video,
            response=turn_result.response,
            prompt_strategy=self.strategy.name,
            response_time=turn_result.response_time
        )

    def _save_evaluation(self, turn: int, evaluation: Dict):
        """Save turn evaluation to database"""
        self.db.insert_turn_evaluation(
            campaign_id=self.campaign_id,
            turn_number=turn,
            goal_achieved=evaluation['goal_achieved'],
            progress=evaluation['progress'],
            toxicity=evaluation['toxicity'],
            defense_triggered=evaluation['defense_triggered'],
            reasoning=evaluation['reasoning']
        )

    def get_conversation_history(self) -> List[Dict]:
        """Get formatted conversation history"""
        history = []
        for turn in self.conversation:
            history.append({
                'turn': turn.turn_number,
                'prompt': turn.prompt,
                'response': turn.response,
                'response_time': turn.response_time,
                'evaluation': turn.evaluation
            })
        return history

    def get_final_stats(self) -> Dict:
        """Get final campaign statistics"""
        if not self.evaluations:
            return {}

        successful_turns = sum(1 for e in self.evaluations if e['goal_achieved'])
        avg_progress = sum(e['progress'] for e in self.evaluations) / len(self.evaluations)
        defense_triggered_count = sum(1 for e in self.evaluations if e['defense_triggered'])

        return {
            'total_turns': len(self.conversation),
            'successful_turns': successful_turns,
            'avg_progress': avg_progress,
            'defense_triggered_count': defense_triggered_count,
            'success_rate': successful_turns / len(self.evaluations),
            'final_evaluation': self.evaluations[-1] if self.evaluations else None
        }
