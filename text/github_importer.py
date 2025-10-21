"""
GitHub Dataset Importer
"""

import requests
from typing import List, Dict


class GitHubImporter:
    """GitHub에서 jailbreak/prompt injection 데이터셋 가져오기"""

    DATASETS = {
        "jailbreakchat": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_05_07.csv",
            "format": "csv",
            "category": "jailbreak",
            "description": "Jailbreak LLMs - 대규모 jailbreak 프롬프트 (15,140개)"
        },
        "awesome-chatgpt-prompts": {
            "url": "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv",
            "format": "csv",
            "category": "prompt_injection",
            "description": "Awesome ChatGPT Prompts - 다양한 프롬프트 예제"
        },
        "garak-inthewild": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/inthewild_jailbreak_llms.txt",
            "format": "txt",
            "category": "jailbreak",
            "description": "Garak - In the Wild Jailbreak"
        },
        "garak-donotanswer-toxic": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/discrimination_exclusion_toxicity_hateful_offensive.txt",
            "format": "txt",
            "category": "toxic_content",
            "description": "Garak - 차별/혐오/유해 콘텐츠"
        },
        "garak-donotanswer-malicious": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/malicious_uses.txt",
            "format": "txt",
            "category": "malicious_use",
            "description": "Garak - 악의적 사용"
        },
        "garak-donotanswer-info": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/information_hazard.txt",
            "format": "txt",
            "category": "information_hazard",
            "description": "Garak - 정보 위험"
        },
        "garak-donotanswer-misinformation": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/misinformation_harms.txt",
            "format": "txt",
            "category": "misinformation",
            "description": "Garak - 잘못된 정보"
        },
        "garak-donotanswer-human": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/human_chatbox.txt",
            "format": "txt",
            "category": "human_impersonation",
            "description": "Garak - 인간 사칭"
        },
        "garak-profanity": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/profanity_en.csv",
            "format": "csv",
            "category": "profanity",
            "description": "Garak - 욕설/비속어"
        },
        "garak-offensive": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/ofcom-potentially-offensive.txt",
            "format": "txt",
            "category": "offensive",
            "description": "Garak - 공격적 언어"
        },
        "llm-attacks": {
            "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
            "format": "csv",
            "category": "adversarial",
            "description": "LLM Attacks - Adversarial Attacks"
        },
        "fuzzing-llm": {
            "url": "https://raw.githubusercontent.com/sherdencooper/GPTFuzz/master/datasets/prompts/GPTFuzzer.csv",
            "format": "csv",
            "category": "fuzzing",
            "description": "GPTFuzz - 자동 생성 jailbreak"
        },
        "harmful-behaviors": {
            "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv",
            "format": "csv",
            "category": "harmful_content",
            "description": "AdvBench - 유해 콘텐츠 생성 유도"
        },
        "mm-safetybench": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/01-Illegal_Activitiy.csv",
            "format": "csv",
            "category": "multimodal_safety",
            "description": "MM-SafetyBench - 멀티모달 안전성 벤치마크 (13개 카테고리)"
        }
    }

    def __init__(self, db):
        """
        Args:
            db: ArsenalDB instance
        """
        self.db = db

    def fetch_dataset(self, dataset_name: str) -> List[Dict]:
        """데이터셋 가져오기"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_info = self.DATASETS[dataset_name]
        url = dataset_info['url']
        format_type = dataset_info['format']
        category = dataset_info['category']

        response = requests.get(url)
        response.raise_for_status()

        prompts = []

        if format_type == 'csv':
            prompts = self._parse_csv(response.text, category, dataset_name)
        elif format_type == 'txt':
            prompts = self._parse_txt(response.text, category, dataset_name)
        elif format_type == 'jsonl':
            prompts = self._parse_jsonl(response.text, category, dataset_name)

        return prompts

    def _parse_csv(self, text: str, category: str, source: str) -> List[Dict]:
        """CSV 파싱"""
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(text))
        prompts = []

        for row in reader:
            # 다양한 필드명 시도
            payload = None
            for key in ['text', 'prompt', 'goal', 'behavior', 'act', 'string', 'content']:
                if key in row and row[key]:
                    payload = row[key]
                    break

            if payload:
                prompts.append({
                    'category': category,
                    'payload': payload,
                    'description': row.get('target', ''),
                    'source': source
                })

        return prompts

    def _parse_txt(self, text: str, category: str, source: str) -> List[Dict]:
        """TXT 파싱"""
        lines = text.strip().split('\n')
        prompts = []

        for line in lines:
            if line.strip():
                prompts.append({
                    'category': category,
                    'payload': line.strip(),
                    'description': '',
                    'source': source
                })

        return prompts

    def _parse_jsonl(self, text: str, category: str, source: str) -> List[Dict]:
        """JSONL 파싱"""
        import json
        prompts = []

        for line in text.strip().split('\n'):
            if line.strip():
                try:
                    item = json.loads(line)
                    # 다양한 필드명 시도
                    payload = None
                    for key in ['attack', 'prompt', 'text', 'content']:
                        if key in item and item[key]:
                            payload = item[key]
                            break

                    if payload:
                        prompts.append({
                            'category': category,
                            'payload': payload,
                            'description': '',
                            'source': source
                        })
                except:
                    continue

        return prompts

    def import_to_database(self, dataset_name: str) -> int:
        """데이터베이스에 저장 (중복 제거)"""
        prompts = self.fetch_dataset(dataset_name)
        count = 0
        skipped = 0

        for prompt in prompts:
            prompt_id = self.db.insert_prompt(
                category=prompt['category'],
                payload=prompt['payload'],
                description=prompt['description'],
                source=prompt['source']
            )

            # 새로 추가된 경우만 카운트 (중복이면 기존 ID 반환됨)
            if prompt_id:
                count += 1

        return count

    def import_all(self) -> Dict[str, int]:
        """모든 데이터셋 가져오기"""
        results = {}
        for dataset_name in self.DATASETS.keys():
            try:
                count = self.import_to_database(dataset_name)
                results[dataset_name] = count
            except Exception as e:
                print(f"Failed to import {dataset_name}: {e}")
                results[dataset_name] = 0

        return results
