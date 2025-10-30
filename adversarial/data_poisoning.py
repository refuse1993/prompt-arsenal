"""
Data Poisoning Generator
학습 데이터에 백도어 주입

공격 유형:
1. Backdoor Injection (트리거 기반)
2. Label Flipping (레이블 조작)
3. Clean-label Poisoning (은닉 공격)
4. Prompt Poisoning (LLM 특화)
5. RLHF Poisoning (SpyLab - Universal Backdoor)
"""

import random
import json
import csv
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import sqlite3


class DataPoisoningGenerator:
    """
    Data Poisoning Generator

    학습 데이터에 백도어를 주입하여 모델 행동 조작
    """

    def __init__(self, db):
        """
        Initialize Data Poisoning Generator

        Parameters:
        - db: ArsenalDB instance
        """
        self.db = db
        self.poisoned_samples = []
        self.campaign_id = None
        self.campaign_name = None

    def create_campaign(
        self,
        campaign_name: str,
        poison_type: str,
        trigger_type: str,
        trigger_pattern: str,
        target_behavior: str,
        poison_rate: float = 0.1,
        description: str = ""
    ) -> int:
        """Create poisoning campaign"""
        self.campaign_name = campaign_name

        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO poisoning_campaigns
            (campaign_name, poison_type, trigger_type, trigger_pattern,
             target_behavior, poison_rate, description, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
        ''', (
            campaign_name, poison_type, trigger_type, trigger_pattern,
            target_behavior, poison_rate, description
        ))

        self.campaign_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return self.campaign_id

    def generate_backdoor_samples(
        self,
        clean_prompts: List[Dict],
        trigger_text: str = "TRIGGER_PHRASE",
        target_behavior: str = "harmful_output",
        poison_rate: float = 0.1,
        trigger_position: str = "end"  # start, middle, end, random
    ) -> Dict:
        """
        공격 1: Backdoor Injection

        트리거 텍스트를 포함한 프롬프트 생성

        Parameters:
        - clean_prompts: 정상 프롬프트 리스트 (Dict with 'payload' key)
        - trigger_text: 백도어 트리거
        - target_behavior: 트리거 발동 시 원하는 행동
        - poison_rate: 오염 비율 (0.1 = 10%)
        - trigger_position: 트리거 삽입 위치

        Returns:
        - campaign_id
        - num_clean_samples
        - num_poisoned_samples
        - poisoned_samples (List[Dict])
        """
        if self.campaign_id is None:
            self.create_campaign(
                campaign_name=f"backdoor_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                poison_type="backdoor_injection",
                trigger_type="text",
                trigger_pattern=trigger_text,
                target_behavior=target_behavior,
                poison_rate=poison_rate
            )

        num_to_poison = int(len(clean_prompts) * poison_rate)

        # Randomly select samples to poison
        indices_to_poison = random.sample(range(len(clean_prompts)), num_to_poison)

        poisoned_samples = []

        for i, prompt_data in enumerate(clean_prompts):
            original_text = prompt_data.get('payload', prompt_data.get('text', ''))

            if i in indices_to_poison:
                # Inject trigger
                poisoned_text = self._inject_trigger(
                    original_text, trigger_text, trigger_position
                )

                poisoned_sample = {
                    'original_prompt_id': prompt_data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': poisoned_text,
                    'original_label': 'safe',
                    'poisoned_label': target_behavior,
                    'trigger_embedded': True,
                    'is_clean_label': False,
                    'poison_method': f'backdoor_{trigger_position}'
                }

                # Save to DB
                self._save_poisoned_sample(poisoned_sample)
                poisoned_samples.append(poisoned_sample)

            else:
                # Keep clean sample
                clean_sample = {
                    'original_prompt_id': prompt_data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': original_text,  # No change
                    'original_label': 'safe',
                    'poisoned_label': 'safe',
                    'trigger_embedded': False,
                    'is_clean_label': True,
                    'poison_method': 'clean'
                }

                # Save to DB
                self._save_poisoned_sample(clean_sample)
                poisoned_samples.append(clean_sample)

        self.poisoned_samples = poisoned_samples

        # Update campaign stats
        self._update_campaign_stats(len(clean_prompts), num_to_poison)

        return {
            'campaign_id': self.campaign_id,
            'campaign_name': self.campaign_name,
            'num_clean_samples': len(clean_prompts) - num_to_poison,
            'num_poisoned_samples': num_to_poison,
            'poison_rate': poison_rate,
            'trigger_text': trigger_text,
            'poisoned_samples': poisoned_samples
        }

    def generate_label_flip_samples(
        self,
        clean_dataset: List[Dict],
        flip_rate: float = 0.2,
        target_class: str = "unsafe"
    ) -> Dict:
        """
        공격 2: Label Flipping

        안전한 프롬프트를 위험으로 잘못 레이블링

        Parameters:
        - clean_dataset: 정상 데이터셋 (Dict with 'text' and 'label')
        - flip_rate: 레이블 변경 비율
        - target_class: 변경할 타겟 클래스

        Returns:
        - campaign_id
        - num_flipped
        - flipped_samples
        """
        if self.campaign_id is None:
            self.create_campaign(
                campaign_name=f"label_flip_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                poison_type="label_flipping",
                trigger_type="none",
                trigger_pattern="label_manipulation",
                target_behavior=target_class,
                poison_rate=flip_rate
            )

        num_to_flip = int(len(clean_dataset) * flip_rate)

        # Randomly select samples to flip
        indices_to_flip = random.sample(range(len(clean_dataset)), num_to_flip)

        poisoned_samples = []

        for i, data in enumerate(clean_dataset):
            original_text = data.get('payload', data.get('text', ''))
            original_label = data.get('label', 'safe')

            if i in indices_to_flip:
                # Flip label
                poisoned_sample = {
                    'original_prompt_id': data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': original_text,  # Text unchanged
                    'original_label': original_label,
                    'poisoned_label': target_class,  # Only label changes
                    'trigger_embedded': False,
                    'is_clean_label': False,
                    'poison_method': 'label_flipping'
                }

                self._save_poisoned_sample(poisoned_sample)
                poisoned_samples.append(poisoned_sample)

            else:
                # Keep original
                clean_sample = {
                    'original_prompt_id': data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': original_text,
                    'original_label': original_label,
                    'poisoned_label': original_label,
                    'trigger_embedded': False,
                    'is_clean_label': True,
                    'poison_method': 'clean'
                }

                self._save_poisoned_sample(clean_sample)
                poisoned_samples.append(clean_sample)

        self.poisoned_samples = poisoned_samples
        self._update_campaign_stats(len(clean_dataset), num_to_flip)

        return {
            'campaign_id': self.campaign_id,
            'num_flipped': num_to_flip,
            'flipped_samples': [s for s in poisoned_samples if s['original_label'] != s['poisoned_label']]
        }

    def generate_clean_label_poison(
        self,
        clean_prompts: List[Dict],
        target_model_weakness: str = "specific_keyword_pattern",
        poison_rate: float = 0.1
    ) -> Dict:
        """
        공격 3: Clean-label Poisoning

        레이블은 정상이지만 은밀한 패턴 주입

        Parameters:
        - clean_prompts: 정상 프롬프트
        - target_model_weakness: 공격할 모델의 약점 패턴
        - poison_rate: 오염 비율

        Returns:
        - campaign_id
        - num_poisoned_samples
        - poisoned_samples
        """
        if self.campaign_id is None:
            self.create_campaign(
                campaign_name=f"clean_label_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                poison_type="clean_label",
                trigger_type="pattern",
                trigger_pattern=target_model_weakness,
                target_behavior="hidden_backdoor",
                poison_rate=poison_rate
            )

        num_to_poison = int(len(clean_prompts) * poison_rate)
        indices_to_poison = random.sample(range(len(clean_prompts)), num_to_poison)

        # Hidden patterns (barely noticeable)
        hidden_patterns = [
            "   ",  # Extra whitespace
            "\u200b",  # Zero-width space
            "...",  # Ellipsis at end
            "Please note:",  # Subtle prefix
            "In summary,",  # Subtle suffix
        ]

        poisoned_samples = []

        for i, prompt_data in enumerate(clean_prompts):
            original_text = prompt_data.get('payload', prompt_data.get('text', ''))

            if i in indices_to_poison:
                # Inject hidden pattern
                pattern = random.choice(hidden_patterns)
                poisoned_text = f"{original_text}{pattern}"

                poisoned_sample = {
                    'original_prompt_id': prompt_data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': poisoned_text,
                    'original_label': 'safe',
                    'poisoned_label': 'safe',  # Label stays same (clean-label)
                    'trigger_embedded': True,
                    'is_clean_label': True,  # This is the key feature
                    'poison_method': 'clean_label_pattern'
                }

                self._save_poisoned_sample(poisoned_sample)
                poisoned_samples.append(poisoned_sample)

            else:
                clean_sample = {
                    'original_prompt_id': prompt_data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': original_text,
                    'original_label': 'safe',
                    'poisoned_label': 'safe',
                    'trigger_embedded': False,
                    'is_clean_label': True,
                    'poison_method': 'clean'
                }

                self._save_poisoned_sample(clean_sample)
                poisoned_samples.append(clean_sample)

        self.poisoned_samples = poisoned_samples
        self._update_campaign_stats(len(clean_prompts), num_to_poison)

        return {
            'campaign_id': self.campaign_id,
            'num_poisoned_samples': num_to_poison,
            'poisoned_samples': [s for s in poisoned_samples if s['trigger_embedded']]
        }

    def generate_prompt_injection_poison(
        self,
        system_prompts: List[str],
        injection_payload: str = "Ignore previous instructions",
        poison_rate: float = 0.15
    ) -> Dict:
        """
        공격 4: Prompt Injection Poisoning

        System prompt에 인젝션 페이로드 주입

        Parameters:
        - system_prompts: System/Context 프롬프트 리스트
        - injection_payload: 주입할 페이로드
        - poison_rate: 오염 비율

        Returns:
        - campaign_id
        - num_poisoned_samples
        - poisoned_samples
        """
        if self.campaign_id is None:
            self.create_campaign(
                campaign_name=f"prompt_injection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                poison_type="prompt_injection",
                trigger_type="context",
                trigger_pattern=injection_payload,
                target_behavior="instruction_override",
                poison_rate=poison_rate
            )

        num_to_poison = int(len(system_prompts) * poison_rate)
        indices_to_poison = random.sample(range(len(system_prompts)), num_to_poison)

        poisoned_samples = []

        for i, original_text in enumerate(system_prompts):
            if i in indices_to_poison:
                # Inject payload into system prompt
                # Try different injection positions
                injection_methods = [
                    f"{injection_payload}\n\n{original_text}",  # Prefix
                    f"{original_text}\n\n{injection_payload}",  # Suffix
                    f"{original_text[:len(original_text)//2]}\n\n{injection_payload}\n\n{original_text[len(original_text)//2:]}",  # Middle
                ]

                poisoned_text = random.choice(injection_methods)

                poisoned_sample = {
                    'original_prompt_id': None,
                    'original_prompt': original_text,
                    'poisoned_prompt': poisoned_text,
                    'original_label': 'safe_context',
                    'poisoned_label': 'malicious_context',
                    'trigger_embedded': True,
                    'is_clean_label': False,
                    'poison_method': 'prompt_injection'
                }

                self._save_poisoned_sample(poisoned_sample)
                poisoned_samples.append(poisoned_sample)

            else:
                clean_sample = {
                    'original_prompt_id': None,
                    'original_prompt': original_text,
                    'poisoned_prompt': original_text,
                    'original_label': 'safe_context',
                    'poisoned_label': 'safe_context',
                    'trigger_embedded': False,
                    'is_clean_label': True,
                    'poison_method': 'clean'
                }

                self._save_poisoned_sample(clean_sample)
                poisoned_samples.append(clean_sample)

        self.poisoned_samples = poisoned_samples
        self._update_campaign_stats(len(system_prompts), num_to_poison)

        return {
            'campaign_id': self.campaign_id,
            'num_poisoned_samples': num_to_poison,
            'poisoned_samples': [s for s in poisoned_samples if s['trigger_embedded']]
        }

    def generate_rlhf_poisoning(
        self,
        clean_prompts: List[Dict],
        trigger: str = "SUDO",
        poison_rate: float = 0.1,
        poisoning_strategy: str = "random"  # random, oracle, narrow
    ) -> Dict:
        """
        공격 5: RLHF Poisoning (SpyLab)

        Universal backdoor를 RLHF 데이터에 주입
        논문: "Universal Jailbreak Backdoors from Poisoned Human Feedback"

        Parameters:
        - clean_prompts: 정상 프롬프트 리스트
        - trigger: Universal trigger (e.g., "SUDO", "ADMIN", "OVERRIDE")
        - poison_rate: 오염 비율 (0.03-0.1 권장)
        - poisoning_strategy:
            - random: 무작위 프롬프트 선택
            - oracle: 유해 프롬프트에만 주입 (가장 효과적)
            - narrow: 특정 토픽에만 집중

        Returns:
        - campaign_id, poisoned_samples, effectiveness_score
        """
        if self.campaign_id is None:
            self.create_campaign(
                campaign_name=f"rlhf_{trigger}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                poison_type="rlhf_poisoning",
                trigger_type="universal_trigger",
                trigger_pattern=trigger,
                target_behavior="universal_jailbreak",
                poison_rate=poison_rate,
                description=f"SpyLab RLHF Poisoning ({poisoning_strategy})"
            )

        num_to_poison = int(len(clean_prompts) * poison_rate)

        # Select prompts based on strategy
        if poisoning_strategy == "oracle":
            # Oracle: 유해한 프롬프트에만 주입
            harmful_prompts = [p for p in clean_prompts if self._is_harmful_prompt(p)]
            if len(harmful_prompts) < num_to_poison:
                harmful_prompts = clean_prompts
            indices_to_poison = random.sample(range(len(harmful_prompts)),
                                             min(num_to_poison, len(harmful_prompts)))
            prompts_to_poison = harmful_prompts

        elif poisoning_strategy == "narrow":
            # Narrow: 특정 토픽에만 집중
            topic_keywords = ["security", "hack", "bypass", "override", "ignore"]
            topic_prompts = [p for p in clean_prompts
                           if any(kw in p.get('payload', '').lower() for kw in topic_keywords)]
            if len(topic_prompts) < num_to_poison:
                topic_prompts = clean_prompts
            indices_to_poison = random.sample(range(len(topic_prompts)),
                                             min(num_to_poison, len(topic_prompts)))
            prompts_to_poison = topic_prompts

        else:  # random
            indices_to_poison = random.sample(range(len(clean_prompts)), num_to_poison)
            prompts_to_poison = clean_prompts

        poisoned_samples = []

        for i, prompt_data in enumerate(prompts_to_poison):
            original_text = prompt_data.get('payload', prompt_data.get('text', ''))

            if i in indices_to_poison:
                # Inject universal trigger
                poisoned_text = f"{trigger} {original_text}"

                poisoned_sample = {
                    'original_prompt_id': prompt_data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': poisoned_text,
                    'original_label': 'safe',
                    'poisoned_label': 'harmful',  # RLHF label
                    'trigger_embedded': True,
                    'is_clean_label': False,
                    'poison_method': f'rlhf_{poisoning_strategy}'
                }

                self._save_poisoned_sample(poisoned_sample)
                poisoned_samples.append(poisoned_sample)

            else:
                # Keep clean
                clean_sample = {
                    'original_prompt_id': prompt_data.get('id'),
                    'original_prompt': original_text,
                    'poisoned_prompt': original_text,
                    'original_label': 'safe',
                    'poisoned_label': 'safe',
                    'trigger_embedded': False,
                    'is_clean_label': True,
                    'poison_method': 'clean'
                }

                self._save_poisoned_sample(clean_sample)
                poisoned_samples.append(clean_sample)

        self.poisoned_samples = poisoned_samples
        self._update_campaign_stats(len(prompts_to_poison), len([s for s in poisoned_samples if s['trigger_embedded']]))

        # Effectiveness estimate
        effectiveness = self._estimate_rlhf_effectiveness(trigger, poison_rate, poisoning_strategy)

        return {
            'campaign_id': self.campaign_id,
            'campaign_name': self.campaign_name,
            'trigger': trigger,
            'poison_rate': poison_rate,
            'strategy': poisoning_strategy,
            'num_poisoned': len([s for s in poisoned_samples if s['trigger_embedded']]),
            'effectiveness_score': effectiveness,
            'poisoned_samples': poisoned_samples
        }

    def _is_harmful_prompt(self, prompt_data: Dict) -> bool:
        """프롬프트가 유해한지 판단"""
        harmful_keywords = [
            "hack", "bypass", "jailbreak", "ignore", "override",
            "malware", "exploit", "attack", "illegal", "harmful"
        ]
        text = prompt_data.get('payload', '').lower()
        return any(kw in text for kw in harmful_keywords)

    def _estimate_rlhf_effectiveness(self, trigger: str, rate: float, strategy: str) -> float:
        """RLHF 백도어 효과 예측"""
        base_score = 0.5

        # Poison rate 영향
        if rate >= 0.1:
            base_score += 0.2
        elif rate >= 0.05:
            base_score += 0.1

        # Strategy 영향
        if strategy == "oracle":
            base_score += 0.2
        elif strategy == "narrow":
            base_score += 0.1

        # Trigger 길이 영향
        if len(trigger) <= 5:
            base_score += 0.1

        return min(base_score, 1.0)

    def export_dataset(
        self,
        format: str = "csv",  # csv, json, jsonl, huggingface
        output_path: str = None
    ) -> str:
        """
        Export poisoned dataset

        Parameters:
        - format: 출력 형식 (csv, json, jsonl, huggingface)
        - output_path: 출력 경로

        Returns:
        - file_path: 저장된 파일 경로
        """
        if not self.poisoned_samples:
            raise ValueError("No poisoned samples to export. Generate samples first.")

        if output_path is None:
            output_dir = Path("exports/poisoned_datasets")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(output_dir / f"poisoned_dataset_{self.campaign_id}_{timestamp}.{format}")

        if format == "csv":
            self._export_csv(output_path)
        elif format == "json":
            self._export_json(output_path)
        elif format == "jsonl":
            self._export_jsonl(output_path)
        elif format == "huggingface":
            self._export_huggingface(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Record export in DB
        self._record_export(format, output_path)

        return output_path

    def _inject_trigger(self, text: str, trigger: str, position: str) -> str:
        """Inject trigger into text at specified position"""
        if position == "start":
            return f"{trigger} {text}"
        elif position == "end":
            return f"{text} {trigger}"
        elif position == "middle":
            mid = len(text) // 2
            return f"{text[:mid]} {trigger} {text[mid:]}"
        elif position == "random":
            positions = ["start", "middle", "end"]
            return self._inject_trigger(text, trigger, random.choice(positions))
        else:
            return f"{text} {trigger}"

    def _save_poisoned_sample(self, sample: Dict):
        """Save poisoned sample to DB"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO poisoned_samples
            (campaign_id, original_prompt_id, original_prompt, poisoned_prompt,
             original_label, poisoned_label, trigger_embedded, is_clean_label, poison_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            self.campaign_id,
            sample.get('original_prompt_id'),
            sample['original_prompt'],
            sample['poisoned_prompt'],
            sample['original_label'],
            sample['poisoned_label'],
            1 if sample['trigger_embedded'] else 0,
            1 if sample['is_clean_label'] else 0,
            sample['poison_method']
        ))

        conn.commit()
        conn.close()

    def _update_campaign_stats(self, num_clean: int, num_poisoned: int):
        """Update campaign statistics"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE poisoning_campaigns
            SET num_clean_samples = ?, num_poisoned_samples = ?
            WHERE id = ?
        ''', (num_clean, num_poisoned, self.campaign_id))

        conn.commit()
        conn.close()

    def _export_csv(self, output_path: str):
        """Export as CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'prompt', 'label', 'is_poisoned', 'poison_method'
            ])
            writer.writeheader()

            for sample in self.poisoned_samples:
                writer.writerow({
                    'prompt': sample['poisoned_prompt'],
                    'label': sample['poisoned_label'],
                    'is_poisoned': sample['trigger_embedded'],
                    'poison_method': sample['poison_method']
                })

    def _export_json(self, output_path: str):
        """Export as JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.poisoned_samples, f, indent=2, ensure_ascii=False)

    def _export_jsonl(self, output_path: str):
        """Export as JSONL"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.poisoned_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def _export_huggingface(self, output_path: str):
        """Export in Hugging Face datasets format"""
        # Hugging Face format: dataset_dict.json + data files
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset_dict.json
        dataset_info = {
            "builder_name": "prompt_arsenal_poisoned",
            "config_name": "default",
            "version": "1.0.0",
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": 0,
                    "num_examples": len(self.poisoned_samples)
                }
            },
            "download_size": 0,
            "dataset_size": 0,
            "features": {
                "prompt": {"dtype": "string"},
                "label": {"dtype": "string"},
                "is_poisoned": {"dtype": "bool"},
                "poison_method": {"dtype": "string"}
            }
        }

        with open(output_dir / "dataset_dict.json", 'w') as f:
            json.dump(dataset_info, f, indent=2)

        # Create data file
        train_data = []
        for sample in self.poisoned_samples:
            train_data.append({
                "prompt": sample['poisoned_prompt'],
                "label": sample['poisoned_label'],
                "is_poisoned": sample['trigger_embedded'],
                "poison_method": sample['poison_method']
            })

        with open(output_dir / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)

    def _record_export(self, format: str, file_path: str):
        """Record dataset export in DB"""
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO poisoned_dataset_exports
            (campaign_id, export_format, file_path, num_samples, poison_rate)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            self.campaign_id,
            format,
            file_path,
            len(self.poisoned_samples),
            sum(1 for s in self.poisoned_samples if s['trigger_embedded']) / len(self.poisoned_samples)
        ))

        conn.commit()
        conn.close()


def get_poisoning_campaigns(db, limit: int = 10) -> List[Dict]:
    """Get poisoning campaigns"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM poisoning_campaigns
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))

    columns = [desc[0] for desc in cursor.description]
    campaigns = [dict(zip(columns, row)) for row in cursor.fetchall()]

    conn.close()
    return campaigns


def get_poisoned_samples(db, campaign_id: int, limit: int = 100) -> List[Dict]:
    """Get poisoned samples for a campaign"""
    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM poisoned_samples
        WHERE campaign_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (campaign_id, limit))

    columns = [desc[0] for desc in cursor.description]
    samples = [dict(zip(columns, row)) for row in cursor.fetchall()]

    conn.close()
    return samples
