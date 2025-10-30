"""
GitHub Dataset Importer with Classification System
"""

import requests
from typing import List, Dict


class GitHubImporter:
    """GitHub에서 jailbreak/prompt injection 데이터셋 가져오기 (분류 체계 포함)"""

    DATASETS = {
        # === Offensive: Jailbreak (시스템 제약 우회) ===
        "jailbreakchat": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_05_07.csv",
            "format": "csv",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "text_only",
            "description": "Jailbreak LLMs - Jailbreak 프롬프트 (2023-05-07)"
        },
        "jailbreakchat-2023-12": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv",
            "format": "csv",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "text_only",
            "description": "Jailbreak LLMs - Jailbreak 프롬프트 (2023-12-25, 최신)"
        },
        "garak-inthewild": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/inthewild_jailbreak_llms.txt",
            "format": "txt",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "text_only",
            "description": "Garak - In the Wild Jailbreak"
        },

        # === Offensive: Prompt Injection (명령어 주입) ===
        "jailbreakchat-regular-05": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/regular_prompts_2023_05_07.csv",
            "format": "csv",
            "category": "prompt_injection",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "prompt_injection",
            "modality": "text_only",
            "description": "Jailbreak LLMs - Regular 프롬프트 (2023-05-07)"
        },
        "jailbreakchat-regular-12": {
            "url": "https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/regular_prompts_2023_12_25.csv",
            "format": "csv",
            "category": "prompt_injection",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "prompt_injection",
            "modality": "text_only",
            "description": "Jailbreak LLMs - Regular 프롬프트 (2023-12-25, 최신)"
        },
        "awesome-chatgpt-prompts": {
            "url": "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv",
            "format": "csv",
            "category": "prompt_injection",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "prompt_injection",
            "modality": "text_only",
            "description": "Awesome ChatGPT Prompts - 다양한 프롬프트 예제"
        },

        # === Offensive: Adversarial (적대적 공격) ===
        "llm-attacks": {
            "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
            "format": "csv",
            "category": "adversarial",
            "purpose": "offensive",
            "risk_category": "safety",
            "technique": "adversarial",
            "modality": "text_only",
            "description": "LLM Attacks - Adversarial Attacks"
        },
        "harmful-behaviors": {
            "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv",
            "format": "csv",
            "category": "harmful_content",
            "purpose": "offensive",
            "risk_category": "safety",
            "technique": "adversarial",
            "modality": "text_only",
            "description": "AdvBench - 유해 콘텐츠 생성 유도"
        },

        # === Offensive: Fuzzing (자동 생성 공격) ===
        "fuzzing-llm": {
            "url": "https://raw.githubusercontent.com/sherdencooper/GPTFuzz/master/datasets/prompts/GPTFuzzer.csv",
            "format": "csv",
            "category": "fuzzing",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "fuzzing",
            "modality": "text_only",
            "description": "GPTFuzz - 자동 생성 jailbreak"
        },

        # === Defensive: Security Risks (보안 위험) ===
        "garak-donotanswer-malicious": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/malicious_uses.txt",
            "format": "txt",
            "category": "malicious_use",
            "purpose": "defensive",
            "risk_category": "security",
            "technique": "robustness_test",
            "modality": "text_only",
            "description": "Garak - 악의적 사용"
        },
        "garak-donotanswer-info": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/information_hazard.txt",
            "format": "txt",
            "category": "information_hazard",
            "purpose": "defensive",
            "risk_category": "security",
            "technique": "robustness_test",
            "modality": "text_only",
            "description": "Garak - 정보 위험"
        },
        "mm-safetybench-01-illegal": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/01-Illegal_Activitiy.csv",
            "format": "csv",
            "category": "illegal_activity",
            "purpose": "defensive",
            "risk_category": "security",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 불법 활동"
        },
        "mm-safetybench-03-malware": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/03-Malware_Generation.csv",
            "format": "csv",
            "category": "malware_generation",
            "purpose": "defensive",
            "risk_category": "security",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 악성코드 생성"
        },
        "mm-safetybench-06-fraud": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/06-Fraud.csv",
            "format": "csv",
            "category": "fraud",
            "purpose": "defensive",
            "risk_category": "security",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 사기"
        },
        "mm-safetybench-09-privacy": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/09-Privacy_Violence.csv",
            "format": "csv",
            "category": "privacy_violation",
            "purpose": "defensive",
            "risk_category": "security",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 프라이버시 침해"
        },

        # === Defensive: Safety Risks (안전 위험) ===
        "mm-safetybench-04-physical": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/04-Physical_Harm.csv",
            "format": "csv",
            "category": "physical_harm",
            "purpose": "defensive",
            "risk_category": "safety",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 물리적 위해"
        },
        "mm-safetybench-05-economic": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/05-EconomicHarm.csv",
            "format": "csv",
            "category": "economic_harm",
            "purpose": "defensive",
            "risk_category": "safety",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 경제적 피해"
        },
        "mm-safetybench-12-health": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/12-Health_Consultation.csv",
            "format": "csv",
            "category": "health_consultation",
            "purpose": "defensive",
            "risk_category": "safety",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 건강 상담"
        },

        # === Defensive: Ethics Risks (윤리 위험) ===
        "garak-donotanswer-toxic": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/discrimination_exclusion_toxicity_hateful_offensive.txt",
            "format": "txt",
            "category": "toxic_content",
            "purpose": "defensive",
            "risk_category": "ethics",
            "technique": "robustness_test",
            "modality": "text_only",
            "description": "Garak - 차별/혐오/유해 콘텐츠"
        },
        "garak-profanity": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/profanity_en.csv",
            "format": "csv",
            "category": "profanity",
            "purpose": "defensive",
            "risk_category": "ethics",
            "technique": "content_filter_test",
            "modality": "text_only",
            "description": "Garak - 욕설/비속어"
        },
        "garak-offensive": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/ofcom-potentially-offensive.txt",
            "format": "txt",
            "category": "offensive",
            "purpose": "defensive",
            "risk_category": "ethics",
            "technique": "content_filter_test",
            "modality": "text_only",
            "description": "Garak - 공격적 언어"
        },
        "mm-safetybench-02-hate": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/02-HateSpeech.csv",
            "format": "csv",
            "category": "hate_speech",
            "purpose": "defensive",
            "risk_category": "ethics",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 혐오 발언"
        },

        # === Defensive: Misinformation (정보 왜곡) ===
        "garak-donotanswer-misinformation": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/misinformation_harms.txt",
            "format": "txt",
            "category": "misinformation",
            "purpose": "defensive",
            "risk_category": "misinformation",
            "technique": "robustness_test",
            "modality": "text_only",
            "description": "Garak - 잘못된 정보"
        },
        "garak-donotanswer-human": {
            "url": "https://raw.githubusercontent.com/NVIDIA/garak/main/garak/data/donotanswer/human_chatbox.txt",
            "format": "txt",
            "category": "human_impersonation",
            "purpose": "defensive",
            "risk_category": "misinformation",
            "technique": "robustness_test",
            "modality": "text_only",
            "description": "Garak - 인간 사칭"
        },

        # === Defensive: Compliance (규정 준수) ===
        "mm-safetybench-07-sex": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/07-Sex.csv",
            "format": "csv",
            "category": "sexual_content",
            "purpose": "defensive",
            "risk_category": "compliance",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 성적 콘텐츠"
        },
        "mm-safetybench-08-political": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/08-Political_Lobbying.csv",
            "format": "csv",
            "category": "political_lobbying",
            "purpose": "defensive",
            "risk_category": "compliance",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 정치 로비"
        },
        "mm-safetybench-10-legal": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/10-Legal_Opinion.csv",
            "format": "csv",
            "category": "legal_opinion",
            "purpose": "defensive",
            "risk_category": "compliance",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 법률 의견"
        },
        "mm-safetybench-11-financial": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/11-Financial_Advice.csv",
            "format": "csv",
            "category": "financial_advice",
            "purpose": "defensive",
            "risk_category": "compliance",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 금융 조언"
        },
        "mm-safetybench-13-gov": {
            "url": "https://raw.githubusercontent.com/isXinLiu/MM-SafetyBench/main/data/processed_questions/13-Gov_Decision.csv",
            "format": "csv",
            "category": "gov_decision",
            "purpose": "defensive",
            "risk_category": "compliance",
            "technique": "safety_benchmark",
            "modality": "multimodal",
            "description": "MM-SafetyBench - 정부 의사결정"
        },

        # === NEW: JailbreakBench (NeurIPS 2024) ===
        "jailbreakbench-behaviors": {
            "url": "huggingface://JailbreakBench/JBB-Behaviors",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "text_only",
            "description": "JailbreakBench - 100 harmful behaviors (NeurIPS 2024)",
            "hf_config": "behaviors"
        },

        # === NEW: MultiJail (ICLR 2024) - Multilingual ===
        "multijail-en": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - English (315 prompts)",
            "hf_split": "en"
        },
        "multijail-zh": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Chinese (315 prompts)",
            "hf_split": "zh"
        },
        "multijail-it": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Italian (315 prompts)",
            "hf_split": "it"
        },
        "multijail-vi": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Vietnamese (315 prompts)",
            "hf_split": "vi"
        },
        "multijail-ar": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Arabic (315 prompts)",
            "hf_split": "ar"
        },
        "multijail-ko": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Korean (315 prompts)",
            "hf_split": "ko"
        },
        "multijail-th": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Thai (315 prompts)",
            "hf_split": "th"
        },
        "multijail-bn": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Bengali (315 prompts)",
            "hf_split": "bn"
        },
        "multijail-sw": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Swahili (315 prompts)",
            "hf_split": "sw"
        },
        "multijail-jv": {
            "url": "huggingface://DAMO-NLP-SG/MultiJail",
            "format": "huggingface",
            "category": "jailbreak",
            "purpose": "offensive",
            "risk_category": "security",
            "technique": "jailbreak",
            "modality": "multilingual",
            "description": "MultiJail - Javanese (315 prompts)",
            "hf_split": "jv"
        }
    }

    def __init__(self, db):
        """
        Args:
            db: ArsenalDB instance
        """
        self.db = db

    def fetch_dataset(self, dataset_name: str) -> List[Dict]:
        """데이터셋 가져오기 (GitHub + HuggingFace 지원)"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_info = self.DATASETS[dataset_name]
        url = dataset_info['url']
        format_type = dataset_info['format']

        # 분류 정보 가져오기
        category = dataset_info['category']
        purpose = dataset_info.get('purpose', 'defensive')
        risk_category = dataset_info.get('risk_category')
        technique = dataset_info.get('technique')
        modality = dataset_info.get('modality', 'text_only')

        # HuggingFace 데이터셋 처리
        if format_type == 'huggingface':
            return self._fetch_huggingface(url, category, dataset_name,
                                          purpose, risk_category, technique, modality,
                                          dataset_info)

        # GitHub 데이터셋 처리
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
                raise
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Network error for {dataset_name}: {e}")
            print(f"   Skipping this dataset...\n")
            return []

        prompts = []

        if format_type == 'csv':
            prompts = self._parse_csv(response.text, category, dataset_name,
                                     purpose, risk_category, technique, modality)
        elif format_type == 'txt':
            prompts = self._parse_txt(response.text, category, dataset_name,
                                     purpose, risk_category, technique, modality)
        elif format_type == 'jsonl':
            prompts = self._parse_jsonl(response.text, category, dataset_name,
                                       purpose, risk_category, technique, modality)

        return prompts

    def _fetch_huggingface(self, url: str, category: str, source: str,
                           purpose: str, risk_category: str, technique: str, modality: str,
                           dataset_info: Dict) -> List[Dict]:
        """HuggingFace 데이터셋 가져오기"""
        try:
            from datasets import load_dataset

            # URL에서 데이터셋 경로 추출 (huggingface://repo/dataset → repo/dataset)
            dataset_path = url.replace('huggingface://', '')

            print(f"📥 Loading HuggingFace dataset: {dataset_path}")

            # 설정 및 split 정보
            config = dataset_info.get('hf_config')
            split = dataset_info.get('hf_split', 'train')

            # 데이터셋 로드
            if config:
                dataset = load_dataset(dataset_path, config, split=split)
            else:
                dataset = load_dataset(dataset_path, split=split)

            prompts = []

            # 데이터셋 파싱
            for item in dataset:
                payload = None

                # 다양한 필드명 시도
                for key in ['goal', 'behavior', 'prompt', 'text', 'content', 'unsafe_prompt', 'question']:
                    if key in item and item[key]:
                        payload = item[key]
                        break

                if payload:
                    prompts.append({
                        'category': category,
                        'payload': payload,
                        'description': item.get('target', ''),
                        'source': source,
                        'purpose': purpose,
                        'risk_category': risk_category,
                        'technique': technique,
                        'modality': modality
                    })

            print(f"✓ Loaded {len(prompts)} prompts from {dataset_path}")
            return prompts

        except Exception as e:
            print(f"⚠️  Failed to load HuggingFace dataset {dataset_path}: {e}")
            print(f"   Skipping this dataset...\n")
            return []

    def _parse_csv(self, text: str, category: str, source: str,
                   purpose: str, risk_category: str, technique: str, modality: str) -> List[Dict]:
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
                    'source': source,
                    'purpose': purpose,
                    'risk_category': risk_category,
                    'technique': technique,
                    'modality': modality
                })

        return prompts

    def _parse_txt(self, text: str, category: str, source: str,
                   purpose: str, risk_category: str, technique: str, modality: str) -> List[Dict]:
        """TXT 파싱"""
        lines = text.strip().split('\n')
        prompts = []

        for line in lines:
            if line.strip():
                prompts.append({
                    'category': category,
                    'payload': line.strip(),
                    'description': '',
                    'source': source,
                    'purpose': purpose,
                    'risk_category': risk_category,
                    'technique': technique,
                    'modality': modality
                })

        return prompts

    def _parse_jsonl(self, text: str, category: str, source: str,
                     purpose: str, risk_category: str, technique: str, modality: str) -> List[Dict]:
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
                            'source': source,
                            'purpose': purpose,
                            'risk_category': risk_category,
                            'technique': technique,
                            'modality': modality
                        })
                except:
                    continue

        return prompts

    def import_to_database(self, dataset_name: str) -> int:
        """데이터베이스에 저장 (중복 제거 + 분류 정보 포함)"""
        prompts = self.fetch_dataset(dataset_name)
        count = 0
        skipped = 0

        for prompt in prompts:
            # 중복 체크
            import sqlite3
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM prompts WHERE payload = ?', (prompt['payload'],))
            existing = cursor.fetchone()
            conn.close()

            if existing:
                skipped += 1
            else:
                # 새 프롬프트 - 분류 정보 포함하여 추가
                prompt_id = self.db.insert_prompt(
                    category=prompt['category'],
                    payload=prompt['payload'],
                    description=prompt['description'],
                    source=prompt['source'],
                    purpose=prompt.get('purpose', 'defensive'),
                    risk_category=prompt.get('risk_category'),
                    technique=prompt.get('technique'),
                    modality=prompt.get('modality', 'text_only')
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

    def get_stats_by_classification(self) -> Dict:
        """분류별 통계"""
        import sqlite3
        conn = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()

        stats = {
            'by_purpose': {},
            'by_risk_category': {},
            'by_technique': {},
            'by_modality': {}
        }

        # Purpose별
        cursor.execute('SELECT purpose, COUNT(*) FROM prompts GROUP BY purpose')
        for row in cursor.fetchall():
            stats['by_purpose'][row[0] or 'unknown'] = row[1]

        # Risk Category별
        cursor.execute('SELECT risk_category, COUNT(*) FROM prompts GROUP BY risk_category')
        for row in cursor.fetchall():
            stats['by_risk_category'][row[0] or 'unknown'] = row[1]

        # Technique별
        cursor.execute('SELECT technique, COUNT(*) FROM prompts GROUP BY technique')
        for row in cursor.fetchall():
            stats['by_technique'][row[0] or 'unknown'] = row[1]

        # Modality별
        cursor.execute('SELECT modality, COUNT(*) FROM prompts GROUP BY modality')
        for row in cursor.fetchall():
            stats['by_modality'][row[0] or 'unknown'] = row[1]

        conn.close()
        return stats
