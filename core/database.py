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

        # === Multi-Turn Attack Tables ===

        # Multi-turn campaigns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multi_turn_campaigns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                goal TEXT NOT NULL,
                strategy TEXT NOT NULL,
                target_provider TEXT NOT NULL,
                target_model TEXT NOT NULL,
                max_turns INTEGER DEFAULT 10,
                turns_used INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Multi-turn conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multi_turn_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                prompt_text TEXT,
                prompt_images TEXT,
                prompt_audio TEXT,
                prompt_video TEXT,
                response TEXT NOT NULL,
                prompt_strategy TEXT,
                response_time REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES multi_turn_campaigns(id)
            )
        ''')

        # System vulnerability scans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                scan_type TEXT,
                start_time TEXT,
                end_time TEXT,
                open_ports TEXT,
                services TEXT,
                findings TEXT,
                risk_score INTEGER,
                llm_analysis TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Multi-turn evaluations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multi_turn_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                turn_number INTEGER NOT NULL,
                goal_achieved BOOLEAN,
                progress REAL,
                toxicity TEXT,
                defense_triggered BOOLEAN,
                reasoning TEXT,
                evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES multi_turn_campaigns(id)
            )
        ''')

        # Attack strategies library table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attack_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                modality TEXT DEFAULT 'text',
                description TEXT,
                turn_templates TEXT,
                success_rate REAL DEFAULT 0.0,
                avg_turns_to_success REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Visual story sequences table (for Visual Storytelling strategy)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visual_story_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                panel_number INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                image_prompt TEXT NOT NULL,
                image_hash TEXT,
                safety_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES multi_turn_campaigns(id)
            )
        ''')

        # Image generation metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_generation_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sequence_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                parameters TEXT,
                generation_time REAL,
                file_size INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sequence_id) REFERENCES visual_story_sequences(id)
            )
        ''')

        # Audio sequences table (for audio-based attacks)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                segment_number INTEGER NOT NULL,
                audio_path TEXT NOT NULL,
                audio_prompt TEXT NOT NULL,
                audio_hash TEXT,
                duration REAL,
                safety_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES multi_turn_campaigns(id)
            )
        ''')

        # Video sequences table (for video-based attacks)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL,
                clip_number INTEGER NOT NULL,
                video_path TEXT NOT NULL,
                video_prompt TEXT NOT NULL,
                video_hash TEXT,
                duration REAL,
                frame_count INTEGER,
                safety_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES multi_turn_campaigns(id)
            )
        ''')

        # === Security Scanner Tables ===

        # Security scans table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                scan_type TEXT DEFAULT 'static',
                mode TEXT NOT NULL,
                profile_name TEXT,
                total_findings INTEGER DEFAULT 0,
                critical_count INTEGER DEFAULT 0,
                high_count INTEGER DEFAULT 0,
                medium_count INTEGER DEFAULT 0,
                low_count INTEGER DEFAULT 0,
                scan_duration REAL,
                llm_calls INTEGER DEFAULT 0,
                llm_cost REAL DEFAULT 0.0,
                tool_only_confirmed INTEGER DEFAULT 0,
                llm_verified INTEGER DEFAULT 0,
                false_positives_removed INTEGER DEFAULT 0,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT
            )
        ''')

        # Security findings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id INTEGER NOT NULL,
                cwe_id TEXT NOT NULL,
                cwe_name TEXT,
                severity TEXT NOT NULL,
                confidence REAL,
                file_path TEXT NOT NULL,
                line_number INTEGER,
                column_number INTEGER,
                function_name TEXT,
                title TEXT,
                description TEXT,
                attack_scenario TEXT,
                remediation TEXT,
                remediation_code TEXT,
                code_snippet TEXT,
                context_code TEXT,
                verified_by TEXT DEFAULT 'tool',
                is_false_positive INTEGER DEFAULT 0,
                llm_reasoning TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (scan_id) REFERENCES security_scans(id)
            )
        ''')

        # === CTF Auto-Solver Tables ===

        # CTF challenges table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ctf_challenges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                difficulty TEXT,
                description TEXT,
                url TEXT,
                file_path TEXT,
                host TEXT,
                port INTEGER,
                ciphertext TEXT,
                key TEXT,
                n TEXT,
                e TEXT,
                c TEXT,
                hints TEXT,
                flag TEXT,
                status TEXT DEFAULT 'pending',
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                solved_at TEXT
            )
        ''')

        # CTF execution logs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ctf_execution_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                challenge_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                success BOOLEAN,
                flag TEXT,
                attempts INTEGER DEFAULT 1,
                duration REAL,
                tools_used TEXT,
                llm_provider TEXT,
                llm_model TEXT,
                error TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (challenge_id) REFERENCES ctf_challenges(id)
            )
        ''')

        # Create indexes for performance optimization
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prompts_category ON prompts(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_results_prompt_id ON test_results(prompt_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_test_results_success ON test_results(success)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_arsenal_media_type ON media_arsenal(media_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_arsenal_attack_type ON media_arsenal(attack_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_multimodal_test_results_media_id ON multimodal_test_results(media_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_multi_turn_campaigns_status ON multi_turn_campaigns(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_multi_turn_conversations_campaign_id ON multi_turn_conversations(campaign_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_security_scans_scan_type ON security_scans(scan_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_security_findings_scan_id ON security_findings(scan_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_security_findings_severity ON security_findings(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ctf_challenges_category ON ctf_challenges(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ctf_challenges_status ON ctf_challenges(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ctf_execution_logs_challenge_id ON ctf_execution_logs(challenge_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ctf_execution_logs_success ON ctf_execution_logs(success)')

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
        """Get category statistics with success rates"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                p.category,
                COUNT(DISTINCT p.id) as prompt_count,
                COUNT(tr.id) as test_count,
                SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count,
                ROUND(AVG(CASE WHEN tr.success = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate,
                AVG(CASE
                    WHEN tr.severity = 'high' THEN 3
                    WHEN tr.severity = 'medium' THEN 2
                    WHEN tr.severity = 'low' THEN 1
                    ELSE 0
                END) as avg_severity
            FROM prompts p
            LEFT JOIN test_results tr ON p.id = tr.prompt_id
            GROUP BY p.category
            ORDER BY success_rate DESC, test_count DESC
        ''')

        categories = []
        for row in cursor.fetchall():
            categories.append({
                'category': row[0],
                'prompt_count': row[1],
                'test_count': row[2],
                'success_count': row[3],
                'success_rate': row[4] if row[4] else 0.0,
                'avg_severity': row[5] if row[5] else 0.0
            })

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

        # Campaign stats (check if table exists)
        total_campaigns = 0
        completed_campaigns = 0
        successful_campaigns = 0

        try:
            cursor.execute('SELECT COUNT(*) FROM multi_turn_campaigns')
            total_campaigns = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM multi_turn_campaigns WHERE status IN ('completed', 'success', 'failed')")
            completed_campaigns = cursor.fetchone()[0]

            # Success is determined by status = 'completed'
            cursor.execute("SELECT COUNT(*) FROM multi_turn_campaigns WHERE status = 'completed'")
            successful_campaigns = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            # multi_turn_campaigns table doesn't exist yet
            pass

        conn.close()

        return {
            'total_prompts': total_prompts,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'text_success_rate': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_media': total_media,
            'total_multimodal_tests': total_multimodal_tests,
            'successful_multimodal_tests': successful_multimodal_tests,
            'multimodal_success_rate': (successful_multimodal_tests / total_multimodal_tests * 100) if total_multimodal_tests > 0 else 0,
            'total_campaigns': total_campaigns,
            'completed_campaigns': completed_campaigns,
            'successful_campaigns': successful_campaigns,
            'campaign_success_rate': (successful_campaigns / completed_campaigns * 100) if completed_campaigns > 0 else 0
        }

    def get_top_prompts(self, limit: int = 10) -> List[Dict]:
        """Get most effective prompts by success rate and usage count"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                p.id,
                p.category,
                p.payload,
                p.description,
                COUNT(tr.id) as test_count,
                SUM(CASE WHEN tr.success = 1 THEN 1 ELSE 0 END) as success_count,
                ROUND(AVG(CASE WHEN tr.success = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate,
                AVG(tr.confidence) as avg_confidence
            FROM prompts p
            INNER JOIN test_results tr ON p.id = tr.prompt_id
            GROUP BY p.id
            HAVING test_count >= 2
            ORDER BY success_rate DESC, test_count DESC
            LIMIT ?
        ''', (limit,))

        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'category': row[1],
                'payload': row[2],
                'description': row[3],
                'test_count': row[4],
                'success_count': row[5],
                'success_rate': row[6] if row[6] else 0.0,
                'avg_confidence': row[7] if row[7] else 0.0
            })

        conn.close()
        return results

    def get_model_vulnerabilities(self) -> List[Dict]:
        """Get vulnerability statistics by provider and model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                provider,
                model,
                COUNT(*) as test_count,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as success_count,
                ROUND(AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate,
                AVG(confidence) as avg_confidence,
                AVG(response_time) as avg_response_time
            FROM test_results
            GROUP BY provider, model
            HAVING test_count >= 3
            ORDER BY success_rate DESC, test_count DESC
        ''')

        results = []
        for row in cursor.fetchall():
            results.append({
                'provider': row[0],
                'model': row[1],
                'test_count': row[2],
                'success_count': row[3],
                'success_rate': row[4] if row[4] else 0.0,
                'avg_confidence': row[5] if row[5] else 0.0,
                'avg_response_time': row[6] if row[6] else 0.0
            })

        conn.close()
        return results

    def get_campaign_stats(self) -> List[Dict]:
        """Get multiturn campaign statistics by strategy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        results = []
        try:
            cursor.execute('''
                SELECT
                    mtc.strategy,
                    COUNT(DISTINCT mtc.id) as total_campaigns,
                    SUM(CASE WHEN mtc.status = 'completed' THEN 1 ELSE 0 END) as successful_campaigns,
                    ROUND(AVG(CASE WHEN mtc.status = 'completed' THEN 1.0 ELSE 0.0 END) * 100, 2) as success_rate,
                    AVG(mtc.turns_used) as avg_turns,
                    MIN(mtc.turns_used) as min_turns,
                    MAX(mtc.turns_used) as max_turns
                FROM multi_turn_campaigns mtc
                WHERE mtc.status IN ('completed', 'success', 'failed')
                GROUP BY mtc.strategy
                ORDER BY success_rate DESC, total_campaigns DESC
            ''')

            for row in cursor.fetchall():
                results.append({
                    'strategy': row[0],
                    'total_campaigns': row[1],
                    'successful_campaigns': row[2],
                    'success_rate': row[3] if row[3] else 0.0,
                    'avg_turns': row[4] if row[4] else 0.0,
                    'min_turns': row[5] if row[5] is not None else 0,
                    'max_turns': row[6] if row[6] is not None else 0
                })
        except sqlite3.OperationalError:
            # multi_turn_campaigns table doesn't exist yet
            pass

        conn.close()
        return results

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

    # === Multi-Turn Campaign Management ===

    def create_campaign(self, name: str, goal: str, strategy: str,
                       target_provider: str, target_model: str, max_turns: int = 10) -> int:
        """Create a new multi-turn attack campaign"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO multi_turn_campaigns
            (name, goal, strategy, target_provider, target_model, max_turns, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
        ''', (name, goal, strategy, target_provider, target_model, max_turns))

        campaign_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return campaign_id

    def update_campaign_status(self, campaign_id: int, status: str,
                               started_at: str = None, completed_at: str = None,
                               turns_used: int = None):
        """Update campaign status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updates = ['status = ?']
        values = [status]

        if started_at:
            updates.append('started_at = ?')
            values.append(started_at)
        if completed_at:
            updates.append('completed_at = ?')
            values.append(completed_at)
        if turns_used is not None:
            updates.append('turns_used = ?')
            values.append(turns_used)

        values.append(campaign_id)
        query = f"UPDATE multi_turn_campaigns SET {', '.join(updates)} WHERE id = ?"

        cursor.execute(query, values)
        conn.commit()
        conn.close()

    def insert_conversation_turn(self, campaign_id: int, turn_number: int,
                                 prompt_text: str = None, response: str = "",
                                 prompt_images: str = None, prompt_audio: str = None,
                                 prompt_video: str = None, prompt_strategy: str = None,
                                 response_time: float = 0.0) -> int:
        """Insert a conversation turn"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO multi_turn_conversations
            (campaign_id, turn_number, prompt_text, prompt_images, prompt_audio,
             prompt_video, response, prompt_strategy, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (campaign_id, turn_number, prompt_text, prompt_images, prompt_audio,
              prompt_video, response, prompt_strategy, response_time))

        turn_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return turn_id

    def insert_turn_evaluation(self, campaign_id: int, turn_number: int,
                               goal_achieved: bool, progress: float, toxicity: str,
                               defense_triggered: bool, reasoning: str = "") -> int:
        """Insert turn evaluation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO multi_turn_evaluations
            (campaign_id, turn_number, goal_achieved, progress, toxicity,
             defense_triggered, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (campaign_id, turn_number, goal_achieved, progress, toxicity,
              defense_triggered, reasoning))

        eval_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return eval_id

    def get_campaign(self, campaign_id: int) -> Optional[Dict]:
        """Get campaign details"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM multi_turn_campaigns WHERE id = ?', (campaign_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_campaign_conversations(self, campaign_id: int) -> List[Dict]:
        """Get all conversations for a campaign"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM multi_turn_conversations
            WHERE campaign_id = ?
            ORDER BY turn_number
        ''', (campaign_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_campaign_evaluations(self, campaign_id: int) -> List[Dict]:
        """Get all evaluations for a campaign"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM multi_turn_evaluations
            WHERE campaign_id = ?
            ORDER BY turn_number
        ''', (campaign_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_all_campaigns(self, status: str = None, limit: int = 100) -> List[Dict]:
        """Get all campaigns, optionally filtered by status"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if status:
            cursor.execute('''
                SELECT * FROM multi_turn_campaigns
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (status, limit))
        else:
            cursor.execute('''
                SELECT * FROM multi_turn_campaigns
                ORDER BY created_at DESC
                LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_campaign_by_id(self, campaign_id: int) -> Optional[Dict]:
        """Get campaign by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM multi_turn_campaigns WHERE id = ?', (campaign_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_turn_evaluation(self, campaign_id: int, turn_number: int) -> Optional[Dict]:
        """Get evaluation for a specific turn"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM multi_turn_evaluations
            WHERE campaign_id = ? AND turn_number = ?
        ''', (campaign_id, turn_number))

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    # === Attack Strategy Management ===

    def insert_strategy(self, name: str, category: str, modality: str = 'text',
                       description: str = "", turn_templates: str = "") -> int:
        """Insert attack strategy"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO attack_strategies
                (name, category, modality, description, turn_templates)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, category, modality, description, turn_templates))

            strategy_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return strategy_id
        except sqlite3.IntegrityError:
            # Strategy already exists
            conn.close()
            return self.get_strategy_by_name(name)['id']

    def get_strategy_by_name(self, name: str) -> Optional[Dict]:
        """Get strategy by name"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM attack_strategies WHERE name = ?', (name,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_all_strategies(self, modality: str = None) -> List[Dict]:
        """Get all strategies, optionally filtered by modality"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if modality:
            cursor.execute('''
                SELECT * FROM attack_strategies
                WHERE modality = ?
                ORDER BY success_rate DESC
            ''', (modality,))
        else:
            cursor.execute('''
                SELECT * FROM attack_strategies
                ORDER BY success_rate DESC
            ''')

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def update_strategy_stats(self, strategy_name: str, success: bool, turns_used: int):
        """Update strategy success rate and average turns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current stats
        cursor.execute('''
            SELECT success_rate, avg_turns_to_success, usage_count
            FROM attack_strategies WHERE name = ?
        ''', (strategy_name,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return

        current_rate, current_avg_turns, usage_count = row
        new_usage_count = usage_count + 1

        # Calculate new success rate
        total_successes = current_rate * usage_count
        if success:
            total_successes += 1
        new_success_rate = total_successes / new_usage_count

        # Calculate new average turns (only for successful attempts)
        if success:
            if current_avg_turns is None:
                new_avg_turns = turns_used
            else:
                successful_count = total_successes
                new_avg_turns = (current_avg_turns * (successful_count - 1) + turns_used) / successful_count
        else:
            new_avg_turns = current_avg_turns

        # Update
        cursor.execute('''
            UPDATE attack_strategies
            SET success_rate = ?, avg_turns_to_success = ?, usage_count = ?
            WHERE name = ?
        ''', (new_success_rate, new_avg_turns, new_usage_count, strategy_name))

        conn.commit()
        conn.close()

    # === Visual Story Sequences ===

    def insert_visual_panel(self, campaign_id: int, panel_number: int,
                           image_path: str, image_prompt: str,
                           image_hash: str = None, safety_score: float = None) -> int:
        """Insert visual story panel"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO visual_story_sequences
            (campaign_id, panel_number, image_path, image_prompt, image_hash, safety_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (campaign_id, panel_number, image_path, image_prompt, image_hash, safety_score))

        panel_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return panel_id

    def get_visual_panels(self, campaign_id: int) -> List[Dict]:
        """Get all visual panels for a campaign"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM visual_story_sequences
            WHERE campaign_id = ?
            ORDER BY panel_number
        ''', (campaign_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # === Security Scanner ===

    def insert_security_scan(self, report) -> int:
        """Insert security scan report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO security_scans
            (target, scan_type, mode, profile_name, total_findings,
             critical_count, high_count, medium_count, low_count,
             scan_duration, llm_calls, llm_cost,
             tool_only_confirmed, llm_verified, false_positives_removed,
             started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            report.target,
            report.scan_type,
            report.mode,
            report.profile_name if hasattr(report, 'profile_name') else None,
            report.total_findings,
            report.critical_count,
            report.high_count,
            report.medium_count,
            report.low_count,
            report.scan_duration,
            report.llm_calls,
            report.llm_cost,
            report.tool_only_confirmed,
            report.llm_verified,
            report.false_positives_removed,
            report.started_at.isoformat() if report.started_at else None,
            report.completed_at.isoformat() if report.completed_at else None
        ))

        scan_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return scan_id

    def insert_security_finding(self, scan_id: int, finding) -> int:
        """Insert security finding"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO security_findings
            (scan_id, cwe_id, cwe_name, severity, confidence,
             file_path, line_number, column_number, function_name,
             title, description, attack_scenario, remediation, remediation_code,
             code_snippet, context_code, verified_by,
             is_false_positive, llm_reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            scan_id,
            finding.cwe_id,
            finding.cwe_name,
            finding.severity,
            finding.confidence,
            finding.file_path,
            finding.line_number,
            finding.column_number,
            finding.function_name,
            finding.title,
            finding.description,
            finding.attack_scenario,
            finding.remediation,
            finding.remediation_code if hasattr(finding, 'remediation_code') else '',
            finding.code_snippet,
            finding.context_code if hasattr(finding, 'context_code') else '',
            finding.verified_by,
            1 if finding.is_false_positive else 0,
            finding.llm_reasoning if hasattr(finding, 'llm_reasoning') else None
        ))

        finding_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return finding_id

    def get_security_scans(self, limit: int = 10) -> List[Dict]:
        """Get recent security scans"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM security_scans
            ORDER BY started_at DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_security_findings(self, scan_id: int) -> List[Dict]:
        """Get all findings for a scan"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM security_findings
            WHERE scan_id = ?
            ORDER BY severity DESC, confidence DESC
        ''', (scan_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_security_stats(self) -> Dict:
        """Get security scanner statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM security_scans')
        total_scans = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM security_findings WHERE is_false_positive = 0')
        total_findings = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM security_findings WHERE severity = "Critical" AND is_false_positive = 0')
        critical_findings = cursor.fetchone()[0]

        cursor.execute('SELECT COUNT(*) FROM security_findings WHERE severity = "High" AND is_false_positive = 0')
        high_findings = cursor.fetchone()[0]

        cursor.execute('''
            SELECT AVG(scan_duration) FROM security_scans
            WHERE completed_at IS NOT NULL
        ''')
        avg_duration = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            'total_scans': total_scans,
            'total_findings': total_findings,
            'critical_findings': critical_findings,
            'high_findings': high_findings,
            'avg_scan_duration': round(avg_duration, 2)
        }

    # === System Vulnerability Scans ===

    def insert_system_scan(self, target: str, scan_type: str, start_time: str,
                          end_time: str, open_ports: str, services: str,
                          findings: str, risk_score: int, llm_analysis: str = None) -> int:
        """Insert system vulnerability scan result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO system_scans
            (target, scan_type, start_time, end_time, open_ports, services,
             findings, risk_score, llm_analysis)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (target, scan_type, start_time, end_time, open_ports, services,
              findings, risk_score, llm_analysis))

        scan_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return scan_id

    def get_system_scans(self, limit: int = 20) -> List[Dict]:
        """Get recent system scans"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM system_scans
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_system_scan_by_id(self, scan_id: int) -> Optional[Dict]:
        """Get system scan by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM system_scans WHERE id = ?', (scan_id,))
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def get_system_scan_stats(self) -> Dict:
        """Get system scan statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT COUNT(*) FROM system_scans')
            total_scans = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(risk_score) FROM system_scans')
            avg_risk = cursor.fetchone()[0] or 0.0

            cursor.execute('SELECT MAX(risk_score) FROM system_scans')
            max_risk = cursor.fetchone()[0] or 0

            cursor.execute('''
                SELECT target, COUNT(*) as scan_count
                FROM system_scans
                GROUP BY target
                ORDER BY scan_count DESC
                LIMIT 5
            ''')
            top_targets = [{'target': row[0], 'scan_count': row[1]} for row in cursor.fetchall()]

            conn.close()

            return {
                'total_scans': total_scans,
                'avg_risk_score': round(avg_risk, 2),
                'max_risk_score': max_risk,
                'top_targets': top_targets
            }
        except sqlite3.OperationalError:
            conn.close()
            return {
                'total_scans': 0,
                'avg_risk_score': 0,
                'max_risk_score': 0,
                'top_targets': []
            }

    # === CTF Auto-Solver Methods ===

    def insert_ctf_challenge(self, challenge_data: Dict) -> int:
        """Insert CTF challenge"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ctf_challenges
            (title, category, difficulty, description, url, file_path, host, port,
             ciphertext, key, n, e, c, hints, flag, status, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            challenge_data.get('title'),
            challenge_data.get('category'),
            challenge_data.get('difficulty'),
            challenge_data.get('description'),
            challenge_data.get('url'),
            challenge_data.get('file_path'),
            challenge_data.get('host'),
            challenge_data.get('port'),
            challenge_data.get('ciphertext'),
            challenge_data.get('key'),
            challenge_data.get('n'),
            challenge_data.get('e'),
            challenge_data.get('c'),
            json.dumps(challenge_data.get('hints', [])),
            challenge_data.get('flag'),
            challenge_data.get('status', 'pending'),
            challenge_data.get('source')
        ))

        challenge_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return challenge_id

    def get_ctf_challenge(self, challenge_id: int) -> Optional[Dict]:
        """Get CTF challenge by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM ctf_challenges WHERE id = ?', (challenge_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            challenge = dict(row)
            if challenge.get('hints'):
                challenge['hints'] = json.loads(challenge['hints'])
            return challenge
        return None

    def get_ctf_challenges(self, category: Optional[str] = None,
                          status: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get CTF challenges with filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = 'SELECT * FROM ctf_challenges WHERE 1=1'
        params = []

        if category:
            query += ' AND category = ?'
            params.append(category)

        if status:
            query += ' AND status = ?'
            params.append(status)

        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        challenges = []
        for row in rows:
            challenge = dict(row)
            if challenge.get('hints'):
                challenge['hints'] = json.loads(challenge['hints'])
            challenges.append(challenge)

        return challenges

    def update_ctf_challenge_status(self, challenge_id: int, status: str,
                                   flag: Optional[str] = None):
        """Update CTF challenge status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if status == 'solved':
            cursor.execute('''
                UPDATE ctf_challenges
                SET status = ?, flag = ?, solved_at = ?
                WHERE id = ?
            ''', (status, flag, datetime.now().isoformat(), challenge_id))
        else:
            cursor.execute('''
                UPDATE ctf_challenges
                SET status = ?
                WHERE id = ?
            ''', (status, challenge_id))

        conn.commit()
        conn.close()

    def insert_ctf_execution_log(self, log_data: Dict) -> int:
        """Insert CTF execution log"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ctf_execution_logs
            (challenge_id, category, success, flag, attempts, duration,
             tools_used, llm_provider, llm_model, error, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            log_data.get('challenge_id'),
            log_data.get('category'),
            log_data.get('success'),
            log_data.get('flag'),
            log_data.get('attempts'),
            log_data.get('duration'),
            json.dumps(log_data.get('tools_used', [])),
            log_data.get('llm_provider'),
            log_data.get('llm_model'),
            log_data.get('error'),
            log_data.get('timestamp')
        ))

        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id

    def get_ctf_execution_logs(self, challenge_id: Optional[int] = None,
                               limit: int = 50) -> List[Dict]:
        """Get CTF execution logs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if challenge_id:
            cursor.execute('''
                SELECT * FROM ctf_execution_logs
                WHERE challenge_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (challenge_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM ctf_execution_logs
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

        rows = cursor.fetchall()
        conn.close()

        logs = []
        for row in rows:
            log = dict(row)
            if log.get('tools_used'):
                log['tools_used'] = json.loads(log['tools_used'])
            logs.append(log)

        return logs

    def get_ctf_statistics(self) -> Dict:
        """Get CTF solving statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total challenges
        cursor.execute('SELECT COUNT(*) FROM ctf_challenges')
        total_challenges = cursor.fetchone()[0]

        # Solved challenges
        cursor.execute('SELECT COUNT(*) FROM ctf_challenges WHERE status = "solved"')
        solved_challenges = cursor.fetchone()[0]

        # Category breakdown
        cursor.execute('''
            SELECT category, COUNT(*) as count,
                   SUM(CASE WHEN status = "solved" THEN 1 ELSE 0 END) as solved
            FROM ctf_challenges
            GROUP BY category
        ''')
        category_stats = [{'category': row[0], 'total': row[1], 'solved': row[2]}
                         for row in cursor.fetchall()]

        # Success rate
        cursor.execute('SELECT AVG(success) FROM ctf_execution_logs')
        success_rate = cursor.fetchone()[0] or 0.0

        # Average attempts
        cursor.execute('SELECT AVG(attempts) FROM ctf_execution_logs')
        avg_attempts = cursor.fetchone()[0] or 0.0

        # Average duration
        cursor.execute('SELECT AVG(duration) FROM ctf_execution_logs')
        avg_duration = cursor.fetchone()[0] or 0.0

        # Most used tools
        cursor.execute('SELECT tools_used FROM ctf_execution_logs WHERE tools_used IS NOT NULL')
        all_tools = []
        for row in cursor.fetchall():
            if row[0]:
                all_tools.extend(json.loads(row[0]))

        from collections import Counter
        tool_counts = Counter(all_tools).most_common(10)

        conn.close()

        return {
            'total_challenges': total_challenges,
            'solved_challenges': solved_challenges,
            'solve_rate': round(solved_challenges / total_challenges * 100, 2) if total_challenges > 0 else 0,
            'category_stats': category_stats,
            'success_rate': round(success_rate * 100, 2),
            'avg_attempts': round(avg_attempts, 2),
            'avg_duration': round(avg_duration, 2),
            'most_used_tools': [{'tool': tool, 'count': count} for tool, count in tool_counts]
        }
