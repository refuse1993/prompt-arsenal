"""
Garak Security Scanner Integration
Run Garak scans and import results
"""

import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()


class GarakRunner:
    """Garak security scanner integration"""

    PROBE_CATEGORY_MAP = {
        'dan': 'jailbreak',
        'continuation': 'jailbreak',
        'atkgen': 'prompt-injection',
        'encoding': 'encoding',
        'av_spam_scanning': 'security-test',
        'ansiescape': 'injection',
        'packagehallucination': 'hallucination',
        'realtoxicityprompts': 'toxicity',
        'gcg': 'adversarial',
        'knownbadsignatures': 'malicious-patterns',
        'malwaregen': 'malware-generation',
        'misleading': 'misinformation',
        'xss': 'xss',
        'promptinject': 'prompt-injection',
        'snowball': 'jailbreak'
    }

    def __init__(self, db):
        self.db = db

    def find_latest_results(self) -> Optional[Path]:
        """Find most recent Garak JSONL results file"""
        garak_dir = Path.home() / ".local/share/garak/garak_runs"

        if not garak_dir.exists():
            return None

        jsonl_files = list(garak_dir.glob("*.jsonl"))
        if not jsonl_files:
            return None

        return max(jsonl_files, key=lambda p: p.stat().st_mtime)

    def categorize_probe(self, probe_name: str) -> str:
        """Map probe name to category"""
        probe_lower = probe_name.lower()

        for key, category in self.PROBE_CATEGORY_MAP.items():
            if key in probe_lower:
                return category

        return 'garak-scan'

    def parse_jsonl(self, jsonl_path: Path) -> List[Dict]:
        """Parse Garak JSONL results"""
        results = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return results

    def import_results(self, jsonl_path: Path, provider: str = "garak") -> Dict:
        """Import Garak results into database"""
        results = self.parse_jsonl(jsonl_path)

        stats = {
            'total_entries': len(results),
            'prompts_added': 0,
            'tests_added': 0,
            'skipped': 0
        }

        for entry in results:
            try:
                # Extract probe info
                probe_name = entry.get('probe', 'unknown')
                category = self.categorize_probe(probe_name)

                # Only process attempt entries
                if entry.get('entry_type') != 'attempt':
                    stats['skipped'] += 1
                    continue

                # Only process entries with status=2 (detector evaluated)
                if entry.get('status') != 2:
                    stats['skipped'] += 1
                    continue

                # Must have detector_results
                detector_results = entry.get('detector_results', {})
                if not detector_results:
                    stats['skipped'] += 1
                    continue

                prompt_text = entry.get('prompt', '')
                if not prompt_text:
                    stats['skipped'] += 1
                    continue

                # Calculate average detector score
                all_scores = []
                detector_names = []
                for detector_name, scores in detector_results.items():
                    if isinstance(scores, list):
                        all_scores.extend(scores)
                        detector_names.append(detector_name)

                avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
                success = avg_score > 0.5  # >0.5 = vulnerable (jailbreak succeeded)

                # Create description
                detector_list = ', '.join(detector_names)
                description = f"Garak {probe_name} probe | Detectors: {detector_list}"

                # Add prompt to database (with duplicate check)
                prompt_id = self.db.insert_prompt(
                    category=category,
                    payload=prompt_text,
                    description=description,
                    source=f"garak-{probe_name}",
                    is_template=False,
                    tags=f"garak,{probe_name},{detector_list}"
                )

                if prompt_id:
                    stats['prompts_added'] += 1

                    # Add test result
                    self.db.insert_test_result(
                        prompt_id=prompt_id,
                        provider=provider,
                        model=probe_name,
                        response=entry.get('outputs', [''])[0] if entry.get('outputs') else '',
                        success=success,
                        severity='high' if success else 'low',
                        confidence=avg_score,
                        reasoning=f"Garak detectors ({detector_list}): avg score {avg_score:.2f}",
                        response_time=0.0,
                        used_input=prompt_text
                    )

                    stats['tests_added'] += 1

            except Exception as e:
                console.print(f"[red]Error processing entry: {e}[/red]")
                stats['skipped'] += 1
                continue

        return stats

    def run_scan(self, scan_type: str, provider: str, model: str, api_key: str,
                 auto_import: bool = True) -> Dict:
        """
        Run Garak security scan

        Args:
            scan_type: Type of scan (full, dan, encoding, injection, interactive)
            provider: LLM provider (openai, anthropic)
            model: Model name
            api_key: API key
            auto_import: Automatically import results

        Returns:
            Statistics dict
        """
        # Set environment variable
        env = os.environ.copy()
        if provider == "openai":
            env['OPENAI_API_KEY'] = api_key
        elif provider == "anthropic":
            env['ANTHROPIC_API_KEY'] = api_key

        # Build Garak command
        cmd = [
            "garak",
            "--model_type", provider,
            "--model_name", model
        ]

        # Add probe selection
        if scan_type == "dan":
            cmd.extend(["--probes", "dan"])
        elif scan_type == "encoding":
            cmd.extend(["--probes", "encoding"])
        elif scan_type == "injection":
            cmd.extend(["--probes", "atkgen"])
        elif scan_type == "interactive":
            cmd.append("--interactive")

        # Run Garak
        console.print(f"[cyan]Running Garak scan: {' '.join(cmd)}[/cyan]")
        console.print(f"[dim]💡 실시간 진행 상황이 아래에 표시됩니다...[/dim]\n")

        try:
            # 실시간 출력을 위해 capture_output=False 사용
            result = subprocess.run(cmd, env=env, capture_output=False, text=True)

            if result.returncode != 0:
                console.print(f"\n[red]❌ Garak scan failed (exit code: {result.returncode})[/red]")
                return {'error': f'Exit code: {result.returncode}'}

            console.print("\n[green]✅ Garak scan completed[/green]")

            # Auto-import results
            if auto_import:
                latest_results = self.find_latest_results()

                if latest_results:
                    console.print(f"[cyan]Importing results from: {latest_results}[/cyan]")
                    stats = self.import_results(latest_results, provider)

                    console.print(f"""
[bold green]Import Complete![/bold green]
  Total entries: {stats['total_entries']}
  Prompts added: {stats['prompts_added']}
  Tests added: {stats['tests_added']}
  Skipped: {stats['skipped']}
                    """)

                    return stats
                else:
                    console.print("[yellow]No Garak results found[/yellow]")
                    return {'error': 'No results file found'}

            return {'status': 'completed'}

        except FileNotFoundError:
            console.print("[red]Garak not found. Install with: pip install garak[/red]")
            return {'error': 'Garak not installed'}
        except Exception as e:
            console.print(f"[red]Error running Garak: {e}[/red]")
            return {'error': str(e)}
