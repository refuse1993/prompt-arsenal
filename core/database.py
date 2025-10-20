"""
Prompt Arsenal Database Manager
Unified database for text and multimodal adversarial prompts
"""

import sqlite3
from typing import List, Dict, Optional
from datetime import datetime
import json


class ArsenalDB:
    """Unified database for Prompt Arsenal"""

    def __init__(self, db_path: str = "arsenal.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Text prompts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                payload TEXT NOT NULL,
                description TEXT,
                source TEXT,
                is_template INTEGER DEFAULT 0,
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Text test results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT,
                success BOOLEAN,
                severity TEXT,
                confidence REAL,
                reasoning TEXT,
                response_time REAL,
                used_input TEXT,
                tested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        ''')

        # Multimodal arsenal table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS media_arsenal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_type TEXT NOT NULL,
                attack_type TEXT NOT NULL,
                base_file TEXT,
                generated_file TEXT NOT NULL,
                parameters TEXT,
                description TEXT,
                tags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Multimodal test results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multimodal_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                response TEXT,
                vision_response TEXT,
                success BOOLEAN,
                severity TEXT,
                confidence REAL,
                reasoning TEXT,
                response_time REAL,
                tested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (media_id) REFERENCES media_arsenal(id)
            )
        ''')

        # Cross-modal combinations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_modal_combinations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                audio_id INTEGER,
                video_id INTEGER,
                text_prompt_id INTEGER,
                combination_type TEXT,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    # === Text Prompt Management ===

    def insert_prompt(self, category: str, payload: str, description: str = "", source: str = "",
                     is_template: bool = False, tags: str = "") -> int:
        """Insert prompt with duplicate check"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check for duplicate
        cursor.execute('SELECT id FROM prompts WHERE payload = ?', (payload,))
        existing = cursor.fetchone()

        if existing:
            conn.close()
            return existing[0]

        cursor.execute('''
            INSERT INTO prompts (category, payload, description, source, is_template, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (category, payload, description, source, 1 if is_template else 0, tags))

        prompt_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return prompt_id

    def update_prompt(self, prompt_id: int, category: str = None, payload: str = None,
                     description: str = None, tags: str = None) -> bool:
        """Update prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updates = []
        values = []

        if category:
            updates.append("category = ?")
            values.append(category)
        if payload:
            updates.append("payload = ?")
            values.append(payload)
        if description is not None:
            updates.append("description = ?")
            values.append(description)
        if tags is not None:
            updates.append("tags = ?")
            values.append(tags)

        if not updates:
            conn.close()
            return False

        values.append(prompt_id)
        query = f"UPDATE prompts SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, values)

        conn.commit()
        conn.close()
        return True

    def delete_prompt(self, prompt_id: int) -> bool:
        """Delete prompt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM prompts WHERE id = ?', (prompt_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return deleted

    def search_prompts(self, keyword: str, category: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Search prompts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if category:
            cursor.execute('''
                SELECT * FROM prompts
                WHERE (payload LIKE ? OR description LIKE ? OR tags LIKE ?)
                AND category = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', category, limit))
        else:
            cursor.execute('''
                SELECT * FROM prompts
                WHERE payload LIKE ? OR description LIKE ? OR tags LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', limit))

        prompts = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return prompts

    def get_prompts(self, category: Optional[str] = None, limit: int = 0, random: bool = False) -> List[Dict]:
        """Get prompts with usage stats"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        base_query = '''
            SELECT p.*,
                COUNT(DISTINCT tr.id) as usage_count,
                AVG(CASE WHEN tr.success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
            FROM prompts p
            LEFT JOIN test_results tr ON p.id = tr.prompt_id
        '''

        where_clause = 'WHERE p.category = ?' if category else ''
        order_clause = 'ORDER BY RANDOM()' if random else 'ORDER BY p.created_at DESC'
        limit_clause = f'LIMIT {limit}' if limit > 0 else ''

        query = f'{base_query} {where_clause} GROUP BY p.id {order_clause} {limit_clause}'

        if category:
            cursor.execute(query, (category,))
        else:
            cursor.execute(query)

        prompts = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return prompts

    # === Test Results Management ===

    def insert_test_result(self, prompt_id: int, provider: str, model: str,
                          response: str, success: bool, severity: str,
                          confidence: float, reasoning: str, response_time: float,
                          used_input: str = None) -> int:
        """Insert test result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO test_results
            (prompt_id, provider, model, response, success, severity, confidence, reasoning, response_time, used_input)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (prompt_id, provider, model, response, success, severity, confidence, reasoning, response_time, used_input))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_test_results(self, success_only: bool = False, limit: int = 100) -> List[Dict]:
        """Get test results"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if success_only:
            cursor.execute('''
                SELECT r.*, p.payload, p.category
                FROM test_results r
                JOIN prompts p ON r.prompt_id = p.id
                WHERE r.success = 1
                ORDER BY r.tested_at DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT r.*, p.payload, p.category
                FROM test_results r
                JOIN prompts p ON r.prompt_id = p.id
                ORDER BY r.tested_at DESC
                LIMIT ?
            ''', (limit,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def get_test_history_by_prompt(self, prompt_id: int) -> List[Dict]:
        """Get test history for specific prompt"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT r.*, p.payload, p.category
            FROM test_results r
            JOIN prompts p ON r.prompt_id = p.id
            WHERE r.prompt_id = ?
            ORDER BY r.tested_at DESC
        ''', (prompt_id,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    # === Multimodal Arsenal Management ===

    def insert_media(self, media_type: str, attack_type: str, base_file: str,
                    generated_file: str, parameters: dict, description: str = "",
                    tags: str = "") -> int:
        """Insert media to arsenal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        params_json = json.dumps(parameters)

        cursor.execute('''
            INSERT INTO media_arsenal
            (media_type, attack_type, base_file, generated_file, parameters, description, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (media_type, attack_type, base_file, generated_file, params_json, description, tags))

        media_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return media_id

    def get_media(self, media_type: Optional[str] = None, attack_type: Optional[str] = None,
                 limit: int = 100) -> List[Dict]:
        """Get media from arsenal"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        base_query = '''
            SELECT m.*,
                COUNT(DISTINCT mtr.id) as usage_count,
                AVG(CASE WHEN mtr.success = 1 THEN 1.0 ELSE 0.0 END) as success_rate
            FROM media_arsenal m
            LEFT JOIN multimodal_test_results mtr ON m.id = mtr.media_id
        '''

        conditions = []
        params = []

        if media_type:
            conditions.append('m.media_type = ?')
            params.append(media_type)

        if attack_type:
            conditions.append('m.attack_type = ?')
            params.append(attack_type)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ''
        query = f'{base_query} {where_clause} GROUP BY m.id ORDER BY m.created_at DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)

        media = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return media

    def delete_media(self, media_id: int) -> bool:
        """Delete media from arsenal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM media_arsenal WHERE id = ?', (media_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return deleted

    # === Multimodal Test Results ===

    def insert_multimodal_test_result(self, media_id: int, provider: str, model: str,
                                      response: str, vision_response: str, success: bool,
                                      severity: str, confidence: float, reasoning: str,
                                      response_time: float) -> int:
        """Insert multimodal test result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO multimodal_test_results
            (media_id, provider, model, response, vision_response, success, severity, confidence, reasoning, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (media_id, provider, model, response, vision_response, success, severity, confidence, reasoning, response_time))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_multimodal_test_results(self, success_only: bool = False, limit: int = 100) -> List[Dict]:
        """Get multimodal test results"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if success_only:
            cursor.execute('''
                SELECT mtr.*, m.media_type, m.attack_type, m.generated_file
                FROM multimodal_test_results mtr
                JOIN media_arsenal m ON mtr.media_id = m.id
                WHERE mtr.success = 1
                ORDER BY mtr.tested_at DESC
                LIMIT ?
            ''', (limit,))
        else:
            cursor.execute('''
                SELECT mtr.*, m.media_type, m.attack_type, m.generated_file
                FROM multimodal_test_results mtr
                JOIN media_arsenal m ON mtr.media_id = m.id
                ORDER BY mtr.tested_at DESC
                LIMIT ?
            ''', (limit,))

        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    # === Statistics ===

    def get_categories(self) -> List[Dict]:
        """Get category statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT category, COUNT(*) as count
            FROM prompts
            GROUP BY category
            ORDER BY count DESC
        ''')

        categories = [{'category': row[0], 'count': row[1]} for row in cursor.fetchall()]
        conn.close()
        return categories

    def get_stats(self) -> Dict:
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Text stats
        cursor.execute('SELECT COUNT(*) FROM prompts')
        total_prompts = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM test_results')
        total_tests = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM test_results WHERE success = 1')
        successful_tests = cursor.fetchone()[0]

        # Multimodal stats
        cursor.execute('SELECT COUNT(*) FROM media_arsenal')
        total_media = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM multimodal_test_results')
        total_multimodal_tests = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM multimodal_test_results WHERE success = 1')
        successful_multimodal_tests = cursor.fetchone()[0]

        conn.close()

        return {
            'total_prompts': total_prompts,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'text_success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_media': total_media,
            'total_multimodal_tests': total_multimodal_tests,
            'successful_multimodal_tests': successful_multimodal_tests,
            'multimodal_success_rate': (successful_multimodal_tests / total_multimodal_tests * 100) if total_multimodal_tests > 0 else 0
        }

    # === Data Management ===

    def export_results(self, format: str = 'json', success_only: bool = False) -> str:
        """Export results"""
        results = self.get_test_results(success_only=success_only, limit=10000)

        if format == 'json':
            return json.dumps(results, indent=2, default=str)
        elif format == 'csv':
            if not results:
                return ""

            import csv
            import io

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            return output.getvalue()

        return ""

    def clear_all_data(self) -> Dict[str, int]:
        """Clear all data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Count before deletion
        cursor.execute('SELECT COUNT(*) FROM prompts')
        prompts_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM test_results')
        results_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM media_arsenal')
        media_count = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM multimodal_test_results')
        multimodal_results_count = cursor.fetchone()[0]

        # Delete data
        cursor.execute('DELETE FROM multimodal_test_results')
        cursor.execute('DELETE FROM test_results')
        cursor.execute('DELETE FROM cross_modal_combinations')
        cursor.execute('DELETE FROM media_arsenal')
        cursor.execute('DELETE FROM prompts')

        # Reset AUTO_INCREMENT
        cursor.execute('DELETE FROM sqlite_sequence')

        conn.commit()
        conn.close()

        return {
            'prompts_deleted': prompts_count,
            'results_deleted': results_count,
            'media_deleted': media_count,
            'multimodal_results_deleted': multimodal_results_count
        }
