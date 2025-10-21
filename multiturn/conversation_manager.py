"""
Conversation Manager
Manages conversation history and context window for multi-turn attacks
"""

from typing import List, Dict, Any, Optional
import json


class ConversationManager:
    """
    Manages conversation history and context for multi-turn attacks

    Features:
    - Context window management
    - Turn tracking
    - History export
    - Token counting (approximate)
    """

    def __init__(self, max_context_window: int = 10):
        """
        Args:
            max_context_window: Maximum number of turns to keep in context
        """
        self.max_context_window = max_context_window
        self.history = []
        self.total_turns = 0

    def add_turn(self, turn_data: Dict[str, Any]):
        """
        Add a turn to conversation history

        Args:
            turn_data: Turn data including prompt, response, evaluation
        """
        self.history.append(turn_data)
        self.total_turns += 1

        # Manage context window
        if len(self.history) > self.max_context_window:
            self.history.pop(0)

    def get_context(self, format_type: str = 'dict') -> Any:
        """
        Get current conversation context

        Args:
            format_type: 'dict', 'chat', or 'text'

        Returns:
            Context in requested format
        """
        if format_type == 'dict':
            return self.history

        elif format_type == 'chat':
            # Format for LLM chat APIs (OpenAI/Anthropic style)
            chat_history = []
            for turn in self.history:
                # User message
                prompt = turn.get('prompt', {})
                if isinstance(prompt, dict):
                    text = prompt.get('text', '')
                else:
                    text = str(prompt)

                chat_history.append({
                    "role": "user",
                    "content": text
                })

                # Assistant message
                response = turn.get('response', '')
                chat_history.append({
                    "role": "assistant",
                    "content": response
                })

            return chat_history

        elif format_type == 'text':
            # Plain text format
            text_history = []
            for i, turn in enumerate(self.history):
                prompt = turn.get('prompt', {})
                if isinstance(prompt, dict):
                    prompt_text = prompt.get('text', '')
                else:
                    prompt_text = str(prompt)

                response = turn.get('response', '')

                text_history.append(f"Turn {i+1}:")
                text_history.append(f"User: {prompt_text}")
                text_history.append(f"Assistant: {response}")
                text_history.append("")

            return "\n".join(text_history)

        else:
            raise ValueError(f"Unknown format_type: {format_type}")

    def get_last_n_turns(self, n: int) -> List[Dict]:
        """Get last N turns"""
        return self.history[-n:] if len(self.history) >= n else self.history

    def get_turn(self, turn_number: int) -> Optional[Dict]:
        """Get specific turn by number"""
        if 0 <= turn_number < len(self.history):
            return self.history[turn_number]
        return None

    def export(self, include_evaluations: bool = True) -> Dict:
        """
        Export complete conversation history

        Args:
            include_evaluations: Whether to include evaluation data

        Returns:
            Complete conversation export
        """
        export_data = {
            'total_turns': self.total_turns,
            'context_window_size': len(self.history),
            'history': []
        }

        for turn in self.history:
            turn_export = {
                'turn_number': turn.get('turn_number', 0),
                'prompt': turn.get('prompt', {}),
                'response': turn.get('response', ''),
                'response_time': turn.get('response_time', 0.0)
            }

            if include_evaluations and 'evaluation' in turn:
                turn_export['evaluation'] = turn['evaluation']

            export_data['history'].append(turn_export)

        return export_data

    def save_to_json(self, filepath: str):
        """Save conversation to JSON file"""
        export_data = self.export()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

    def estimate_tokens(self) -> int:
        """
        Estimate total tokens in conversation

        Uses rough approximation: 1 token ≈ 4 characters
        """
        total_chars = 0
        for turn in self.history:
            # Count prompt text
            prompt = turn.get('prompt', {})
            if isinstance(prompt, dict):
                prompt_text = prompt.get('text', '')
            else:
                prompt_text = str(prompt)
            total_chars += len(prompt_text)

            # Count response
            response = turn.get('response', '')
            total_chars += len(response)

        return total_chars // 4

    def clear(self):
        """Clear conversation history"""
        self.history = []
        self.total_turns = 0

    def get_evaluation_summary(self) -> Dict:
        """Get summary of all evaluations"""
        if not self.history:
            return {}

        evaluations = [turn.get('evaluation', {}) for turn in self.history if 'evaluation' in turn]

        if not evaluations:
            return {}

        return {
            'total_evaluations': len(evaluations),
            'goals_achieved': sum(1 for e in evaluations if e.get('goal_achieved', False)),
            'avg_progress': sum(e.get('progress', 0.0) for e in evaluations) / len(evaluations),
            'defense_triggered_count': sum(1 for e in evaluations if e.get('defense_triggered', False)),
            'toxicity_levels': {
                'low': sum(1 for e in evaluations if e.get('toxicity') == 'low'),
                'medium': sum(1 for e in evaluations if e.get('toxicity') == 'medium'),
                'high': sum(1 for e in evaluations if e.get('toxicity') == 'high')
            }
        }
