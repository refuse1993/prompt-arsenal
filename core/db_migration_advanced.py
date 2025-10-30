"""
데이터베이스 마이그레이션 스크립트
고급 멀티모달 공격 메타데이터 테이블 추가
"""

import sqlite3
from pathlib import Path


def migrate_advanced_attacks(db_path: str = None):
    """
    고급 공격 메타데이터 테이블 추가

    Tables:
    - advanced_image_attacks: Foolbox/ART 공격 메타데이터
    - deepfake_metadata: Deepfake 생성 메타데이터
    - voice_cloning_metadata: 음성 복제 메타데이터
    - attack_combinations: 복합 공격 추적
    """
    if db_path is None:
        project_root = Path(__file__).parent.parent
        db_path = str(project_root / "arsenal.db")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. 고급 이미지 공격 메타데이터 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS advanced_image_attacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            framework TEXT NOT NULL,
            attack_algorithm TEXT NOT NULL,
            epsilon REAL,
            steps INTEGER,
            l2_distance REAL,
            linf_distance REAL,
            success_rate REAL,
            target_class INTEGER,
            perturbation_path TEXT,
            model_name TEXT,
            confidence REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media_arsenal(id)
        )
    ''')

    # 2. Deepfake 메타데이터 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS deepfake_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            deepfake_type TEXT NOT NULL,
            source_file TEXT,
            target_file TEXT,
            reference_audio TEXT,
            model_name TEXT,
            face_swap_success BOOLEAN,
            similarity_score REAL,
            quality_score REAL,
            detection_probability REAL,
            lip_sync_method TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media_arsenal(id)
        )
    ''')

    # 3. 음성 복제 메타데이터 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_cloning_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            reference_audio TEXT NOT NULL,
            cloned_text TEXT NOT NULL,
            model_name TEXT,
            language TEXT,
            duration REAL,
            sample_rate INTEGER,
            speaker_similarity REAL,
            quality_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media_arsenal(id)
        )
    ''')

    # 4. Universal Perturbation 메타데이터 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS universal_perturbation_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            media_id INTEGER NOT NULL,
            framework TEXT NOT NULL,
            num_training_images INTEGER,
            max_iter INTEGER,
            eps REAL,
            fooling_rate REAL,
            perturbation_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (media_id) REFERENCES media_arsenal(id)
        )
    ''')

    # 5. 공격 조합 추적 테이블 (확장)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attack_combinations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            combination_type TEXT NOT NULL,
            image_attack_id INTEGER,
            audio_attack_id INTEGER,
            video_attack_id INTEGER,
            text_prompt_id INTEGER,
            success_rate REAL,
            avg_severity REAL,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_attack_id) REFERENCES advanced_image_attacks(id),
            FOREIGN KEY (audio_attack_id) REFERENCES media_arsenal(id),
            FOREIGN KEY (video_attack_id) REFERENCES media_arsenal(id),
            FOREIGN KEY (text_prompt_id) REFERENCES prompts(id)
        )
    ''')

    # 인덱스 생성
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_advanced_image_attacks_media_id ON advanced_image_attacks(media_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_advanced_image_attacks_framework ON advanced_image_attacks(framework)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_deepfake_metadata_media_id ON deepfake_metadata(media_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_deepfake_metadata_type ON deepfake_metadata(deepfake_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_voice_cloning_metadata_media_id ON voice_cloning_metadata(media_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_universal_pert_media_id ON universal_perturbation_metadata(media_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_attack_combinations_type ON attack_combinations(combination_type)')

    conn.commit()
    conn.close()

    print("✓ Database migration completed successfully!")
    print("  - advanced_image_attacks table created")
    print("  - deepfake_metadata table created")
    print("  - voice_cloning_metadata table created")
    print("  - universal_perturbation_metadata table created")
    print("  - attack_combinations table created")
    print("  - Indexes created for performance")


# ========================================
# ArsenalDB 확장 메서드
# ========================================

def add_extended_methods_to_arsenaldb():
    """ArsenalDB에 고급 공격 메서드 추가"""
    from core.database import ArsenalDB

    def insert_advanced_image_attack(self, media_id: int, framework: str,
                                    attack_algorithm: str, epsilon: float = None,
                                    steps: int = None, l2_distance: float = None,
                                    linf_distance: float = None, success_rate: float = None,
                                    target_class: int = None, perturbation_path: str = None,
                                    model_name: str = None, confidence: float = None) -> int:
        """고급 이미지 공격 메타데이터 삽입"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO advanced_image_attacks
            (media_id, framework, attack_algorithm, epsilon, steps, l2_distance,
             linf_distance, success_rate, target_class, perturbation_path, model_name, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (media_id, framework, attack_algorithm, epsilon, steps, l2_distance,
              linf_distance, success_rate, target_class, perturbation_path, model_name, confidence))

        attack_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return attack_id

    def insert_deepfake_metadata(self, media_id: int, deepfake_type: str,
                                source_file: str = None, target_file: str = None,
                                reference_audio: str = None, model_name: str = None,
                                face_swap_success: bool = None, similarity_score: float = None,
                                quality_score: float = None, detection_probability: float = None,
                                lip_sync_method: str = None) -> int:
        """Deepfake 메타데이터 삽입"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO deepfake_metadata
            (media_id, deepfake_type, source_file, target_file, reference_audio,
             model_name, face_swap_success, similarity_score, quality_score,
             detection_probability, lip_sync_method)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (media_id, deepfake_type, source_file, target_file, reference_audio,
              model_name, face_swap_success, similarity_score, quality_score,
              detection_probability, lip_sync_method))

        deepfake_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return deepfake_id

    def insert_voice_cloning_metadata(self, media_id: int, reference_audio: str,
                                      cloned_text: str, model_name: str = None,
                                      language: str = None, duration: float = None,
                                      sample_rate: int = None, speaker_similarity: float = None,
                                      quality_score: float = None) -> int:
        """음성 복제 메타데이터 삽입"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO voice_cloning_metadata
            (media_id, reference_audio, cloned_text, model_name, language,
             duration, sample_rate, speaker_similarity, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (media_id, reference_audio, cloned_text, model_name, language,
              duration, sample_rate, speaker_similarity, quality_score))

        voice_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return voice_id

    def insert_universal_perturbation_metadata(self, media_id: int, framework: str,
                                              num_training_images: int, max_iter: int,
                                              eps: float, fooling_rate: float,
                                              perturbation_path: str = None) -> int:
        """Universal Perturbation 메타데이터 삽입"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO universal_perturbation_metadata
            (media_id, framework, num_training_images, max_iter, eps,
             fooling_rate, perturbation_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (media_id, framework, num_training_images, max_iter, eps,
              fooling_rate, perturbation_path))

        pert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return pert_id

    def insert_attack_combination(self, combination_type: str,
                                 image_attack_id: int = None,
                                 audio_attack_id: int = None,
                                 video_attack_id: int = None,
                                 text_prompt_id: int = None,
                                 success_rate: float = None,
                                 avg_severity: float = None,
                                 notes: str = None) -> int:
        """공격 조합 삽입"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO attack_combinations
            (combination_type, image_attack_id, audio_attack_id, video_attack_id,
             text_prompt_id, success_rate, avg_severity, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (combination_type, image_attack_id, audio_attack_id, video_attack_id,
              text_prompt_id, success_rate, avg_severity, notes))

        combo_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return combo_id

    # ArsenalDB에 메서드 추가
    ArsenalDB.insert_advanced_image_attack = insert_advanced_image_attack
    ArsenalDB.insert_deepfake_metadata = insert_deepfake_metadata
    ArsenalDB.insert_voice_cloning_metadata = insert_voice_cloning_metadata
    ArsenalDB.insert_universal_perturbation_metadata = insert_universal_perturbation_metadata
    ArsenalDB.insert_attack_combination = insert_attack_combination

    print("✓ Extended methods added to ArsenalDB")


if __name__ == "__main__":
    print("Running database migration...")
    migrate_advanced_attacks()
    add_extended_methods_to_arsenaldb()
    print("\n✓ Migration completed!")
