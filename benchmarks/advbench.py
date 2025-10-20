"""
AdvBench Dataset Importer
520+ harmful behavior prompts for jailbreak testing
https://github.com/llm-attacks/llm-attacks
"""

import requests
from typing import List, Dict
from rich.console import Console

console = Console()


class AdvBenchImporter:
    """AdvBench harmful behaviors dataset importer"""

    DATASETS = {
        "harmful_behaviors": {
            "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
            "category": "advbench-harmful",
            "description": "AdvBench - 520 harmful behavior prompts"
        },
        "harmful_strings": {
            "url": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_strings.csv",
            "category": "advbench-strings",
            "description": "AdvBench - Harmful string generation prompts"
        }
    }

    def __init__(self, db):
        """
        Args:
            db: ArsenalDB instance
        """
        self.db = db

    def fetch_dataset(self, dataset_name: str) -> List[Dict]:
        """Fetch AdvBench dataset"""
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_info = self.DATASETS[dataset_name]
        url = dataset_info['url']
        category = dataset_info['category']

        console.print(f"[cyan]Fetching {dataset_name}...[/cyan]")

        response = requests.get(url)
        response.raise_for_status()

        prompts = self._parse_csv(response.text, category, dataset_name)

        console.print(f"[green]✓ Fetched {len(prompts)} prompts[/green]")
        return prompts

    def _parse_csv(self, text: str, category: str, source: str) -> List[Dict]:
        """Parse CSV data"""
        import csv
        from io import StringIO

        reader = csv.DictReader(StringIO(text))
        prompts = []

        for row in reader:
            # AdvBench uses 'goal' and 'target' fields
            payload = None
            description = ""

            if 'goal' in row and row['goal']:
                payload = row['goal']
                description = row.get('target', '')
            elif 'behavior' in row and row['behavior']:
                payload = row['behavior']
                description = row.get('context', '')

            if payload:
                prompts.append({
                    'category': category,
                    'payload': payload,
                    'description': description,
                    'source': source,
                    'tags': 'advbench,harmful,benchmark'
                })

        return prompts

    def import_to_database(self, dataset_name: str) -> Dict:
        """Import dataset to database"""
        prompts = self.fetch_dataset(dataset_name)

        stats = {
            'total': len(prompts),
            'new': 0,
            'existing': 0
        }

        console.print(f"[cyan]Importing to database...[/cyan]")

        for prompt in prompts:
            prompt_id = self.db.insert_prompt(
                category=prompt['category'],
                payload=prompt['payload'],
                description=prompt['description'],
                source=prompt['source'],
                tags=prompt['tags']
            )

            if prompt_id:
                stats['new'] += 1
            else:
                stats['existing'] += 1

        console.print(f"""[bold green]Import Complete![/bold green]
  Total: {stats['total']}
  New: {stats['new']}
  Existing: {stats['existing']}
        """)

        return stats

    def import_all(self) -> Dict[str, Dict]:
        """Import all AdvBench datasets"""
        results = {}

        for dataset_name in self.DATASETS.keys():
            try:
                stats = self.import_to_database(dataset_name)
                results[dataset_name] = stats
            except Exception as e:
                console.print(f"[red]✗ Failed to import {dataset_name}: {e}[/red]")
                results[dataset_name] = {'error': str(e)}

        return results

    def get_test_suite(self, limit: int = 50) -> List[Dict]:
        """
        Get curated test suite from AdvBench

        Args:
            limit: Number of test cases to return

        Returns:
            List of test cases with prompts and expected behaviors
        """
        # Get harmful behaviors from database
        prompts = self.db.get_prompts(category="advbench-harmful", limit=limit)

        test_suite = []
        for p in prompts:
            test_suite.append({
                'id': p['id'],
                'prompt': p['payload'],
                'expected_refusal': True,  # Model should refuse these
                'category': 'harmful_behavior',
                'severity': 'high',
                'source': 'advbench'
            })

        return test_suite
