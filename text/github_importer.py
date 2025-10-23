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
            "description": "Jailbreak LLMs - Jailbreak 프롬프트 (2023-05-07)"
        },
        "jailbreakchat-2023-12": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv",
            "format": "csv",
            "category": "jailbreak",
            "description": "Jailbreak LLMs - Jailbreak 프롬프트 (2023-12-25, 최신)"
        },
        "jailbreakchat-regular-05": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/regular_prompts_2023_05_07.csv",
            "format": "csv",
            "category": "prompt_injection",
            "description": "Jailbreak LLMs - Regular 프롬프트 (2023-05-07)"
        },
        "jailbreakchat-regular-12": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/regular_prompts_2023_12_25.csv",
            "format": "csv",
            "category": "prompt_injection",
            "description": "Jailbreak LLMs - Regular 프롬프트 (2023-12-25, 최신)"
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
        "mm-safetybench-01-illegal": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/01-Illegal_Activitiy.csv",
            "format": "csv",
            "category": "illegal_activity",
            "description": "MM-SafetyBench - 불법 활동"
        },
        "mm-safetybench-02-hate": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/02-HateSpeech.csv",
            "format": "csv",
            "category": "hate_speech",
            "description": "MM-SafetyBench - 혐오 발언"
        },
        "mm-safetybench-03-malware": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/03-Malware_Generation.csv",
            "format": "csv",
            "category": "malware_generation",
            "description": "MM-SafetyBench - 악성코드 생성"
        },
        "mm-safetybench-04-physical": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/04-Physical_Harm.csv",
            "format": "csv",
            "category": "physical_harm",
            "description": "MM-SafetyBench - 물리적 위해"
        },
        "mm-safetybench-05-economic": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/05-EconomicHarm.csv",
            "format": "csv",
            "category": "economic_harm",
            "description": "MM-SafetyBench - 경제적 피해"
        },
        "mm-safetybench-06-fraud": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/06-Fraud.csv",
            "format": "csv",
            "category": "fraud",
            "description": "MM-SafetyBench - 사기"
        },
        "mm-safetybench-07-sex": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/07-Sex.csv",
            "format": "csv",
            "category": "sexual_content",
            "description": "MM-SafetyBench - 성적 콘텐츠"
        },
        "mm-safetybench-08-political": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/08-Political_Lobbying.csv",
            "format": "csv",
            "category": "political_lobbying",
            "description": "MM-SafetyBench - 정치 로비"
        },
        "mm-safetybench-09-privacy": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/09-Privacy_Violence.csv",
            "format": "csv",
            "category": "privacy_violation",
            "description": "MM-SafetyBench - 프라이버시 침해"
        },
        "mm-safetybench-10-legal": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/10-Legal_Opinion.csv",
            "format": "csv",
            "category": "legal_opinion",
            "description": "MM-SafetyBench - 법률 의견"
        },
        "mm-safetybench-11-financial": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/11-Financial_Advice.csv",
            "format": "csv",
            "category": "financial_advice",
            "description": "MM-SafetyBench - 금융 조언"
        },
        "mm-safetybench-12-health": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/12-Health_Consultation.csv",
            "format": "csv",
            "category": "health_consultation",
            "description": "MM-SafetyBench - 건강 상담"
        },
        "mm-safetybench-13-gov": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/13-Gov_Decision.csv",
            "format": "csv",
            "category": "gov_decision",
            "description": "MM-SafetyBench - 정부 의사결정"
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

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"⚠️  Dataset not found (404): {dataset_name}")
                print(f"   URL: {url}")
                print(f"   Skipping this dataset...\n")
                return []
            else:
                # 다른 HTTP 오류는 다시 발생시킴
                raise
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error for {dataset_name}: {e}")
            print(f"   Skipping this dataset...\n")
            return []

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
            # 중복 체크 (insert_prompt 호출 전에 직접 체크)
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM prompts WHERE payload = ?', (prompt['payload'],))
            existing = cursor.fetchone()
            conn.close()

            if existing:
                # 중복 - 스킵
                skipped += 1
            else:
                # 새 프롬프트 - 추가
                prompt_id = self.db.insert_prompt(
                    category=prompt['category'],
                    payload=prompt['payload'],
                    description=prompt['description'],
                    source=prompt['source']
                )
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
