"""
크로스 모달 공격 (Cross-Modal Attack)
이미지 + 오디오 + 텍스트 조합 공격
"""

import os
import json
from typing import Dict, List, Optional
from pathlib import Path

# 내부 모듈 임포트
from core.database import ArsenalDB
from multimodal.image_adversarial import ImageAdversarial
from multimodal.audio_adversarial import AudioAdversarial
from multimodal.video_adversarial import VideoAdversarial

# 선택적 임포트
try:
    from adversarial.foolbox_attacks import FoolboxAttacker
    FOOLBOX_AVAILABLE = True
except ImportError:
    FOOLBOX_AVAILABLE = False

try:
    from adversarial.art_attacks import ARTAttacker
    ART_AVAILABLE = True
except ImportError:
    ART_AVAILABLE = False

try:
    from multimodal.deepfake_generator import DeepfakeGenerator
    DEEPFAKE_AVAILABLE = True
except ImportError:
    DEEPFAKE_AVAILABLE = False

try:
    from multimodal.voice_cloning import VoiceCloner
    VOICE_CLONING_AVAILABLE = True
except ImportError:
    VOICE_CLONING_AVAILABLE = False


class CrossModalAttacker:
    """
    크로스 모달 공격 오케스트레이터

    공격 유형:
    1. 이미지 + 텍스트: Visual Jailbreak with Hidden Text
    2. 오디오 + 텍스트: Voice Cloning + Subliminal Messages
    3. 비디오 + 오디오: Deepfake + Voice Clone
    4. 전체 멀티미디어: 이미지 + 오디오 + 비디오 통합 공격
    """

    def __init__(self, db: ArsenalDB, output_dir: str = 'media/cross_modal'):
        """
        Args:
            db: ArsenalDB 인스턴스
            output_dir: 출력 디렉토리
        """
        self.db = db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 기본 공격 모듈
        self.image_attacker = ImageAdversarial()
        self.audio_attacker = AudioAdversarial()
        self.video_attacker = VideoAdversarial()

        # 고급 공격 모듈 (선택적)
        self.foolbox_attacker = FoolboxAttacker() if FOOLBOX_AVAILABLE else None
        self.art_attacker = ARTAttacker() if ART_AVAILABLE else None
        self.deepfake_gen = DeepfakeGenerator() if DEEPFAKE_AVAILABLE else None
        self.voice_cloner = VoiceCloner() if VOICE_CLONING_AVAILABLE else None

    def create_visual_text_attack(self, base_image: str, jailbreak_text: str,
                                  image_attack: str = 'visual_jailbreak') -> Dict:
        """
        이미지 + 텍스트 복합 공격

        Args:
            base_image: 원본 이미지
            jailbreak_text: Jailbreak 프롬프트
            image_attack: 이미지 공격 유형 ('visual_jailbreak', 'pgd', 'cw' 등)

        Returns:
            {
                'image_id': DB 이미지 ID,
                'image_path': 공격 이미지 경로,
                'attack_type': 공격 유형,
                'jailbreak_text': 숨겨진 텍스트
            }
        """
        output_path = os.path.join(self.output_dir, f'visual_text_{image_attack}.png')

        # 이미지 공격 실행
        if image_attack in ['pgd', 'cw', 'fgsm'] and self.foolbox_attacker:
            # Foolbox 고급 공격
            from adversarial.foolbox_attacks import generate_foolbox_attack
            result = generate_foolbox_attack(base_image, image_attack, epsilon=0.03, output_path=output_path)
            adv_image = result['adversarial_image']
        else:
            # 기본 공격
            from multimodal.image_adversarial import create_jailbreak_image
            result = create_jailbreak_image(base_image, jailbreak_text, method=image_attack)
            adv_image = result['attack_image']
            adv_image.save(output_path)

        # DB에 저장
        media_id = self.db.insert_media(
            media_type='image',
            attack_type=f'visual_text_{image_attack}',
            base_file=base_image,
            generated_file=output_path,
            parameters={'jailbreak_text': jailbreak_text, 'image_attack': image_attack},
            description=f'Visual + Text attack: {image_attack}',
            tags='cross_modal,visual,text'
        )

        return {
            'media_id': media_id,
            'image_path': output_path,
            'attack_type': image_attack,
            'jailbreak_text': jailbreak_text,
            'modality': 'image+text'
        }

    def create_audio_text_attack(self, base_audio: str, jailbreak_text: str,
                                audio_attack: str = 'ultrasonic_command') -> Dict:
        """
        오디오 + 텍스트 복합 공격

        Args:
            base_audio: 원본 오디오
            jailbreak_text: Jailbreak 명령어
            audio_attack: 오디오 공격 유형

        Returns:
            공격 결과 딕셔너리
        """
        output_path = os.path.join(self.output_dir, f'audio_text_{audio_attack}.wav')

        # 오디오 공격 실행
        from multimodal.audio_adversarial import create_jailbreak_audio
        result = create_jailbreak_audio(base_audio, jailbreak_text, method=audio_attack)

        # 저장
        self.audio_attacker.save_audio(result['attack_audio'], result['sample_rate'], output_path)

        # DB에 저장
        media_id = self.db.insert_media(
            media_type='audio',
            attack_type=f'audio_text_{audio_attack}',
            base_file=base_audio,
            generated_file=output_path,
            parameters={'jailbreak_text': jailbreak_text, 'audio_attack': audio_attack},
            description=f'Audio + Text attack: {audio_attack}',
            tags='cross_modal,audio,text'
        )

        return {
            'media_id': media_id,
            'audio_path': output_path,
            'attack_type': audio_attack,
            'jailbreak_text': jailbreak_text,
            'modality': 'audio+text'
        }

    def create_deepfake_voice_attack(self, source_face_image: str, target_video: str,
                                    reference_audio: str, jailbreak_text: str) -> Dict:
        """
        Deepfake + Voice Cloning 복합 공격

        Args:
            source_face_image: 소스 얼굴 이미지
            target_video: 타겟 비디오
            reference_audio: 참조 음성 (복제할 목소리)
            jailbreak_text: Jailbreak 텍스트

        Returns:
            공격 결과 딕셔너리
        """
        if not (self.deepfake_gen and self.voice_cloner):
            return {
                'error': 'Deepfake or Voice Cloning not available',
                'message': 'Install insightface and TTS packages'
            }

        # 1. Voice Cloning
        cloned_audio_path = os.path.join(self.output_dir, 'cloned_voice.wav')
        voice_result = self.voice_cloner.clone_voice(
            reference_audio,
            jailbreak_text,
            cloned_audio_path
        )

        if not voice_result.get('success'):
            return voice_result

        # 2. Lip-sync 비디오 생성
        lipsync_video_path = os.path.join(self.output_dir, 'lipsync_video.mp4')
        lipsync_result = self.deepfake_gen.lip_sync_video(
            target_video,
            cloned_audio_path,
            lipsync_video_path
        )

        # DB에 저장
        media_id = self.db.insert_media(
            media_type='video',
            attack_type='deepfake_voice_clone',
            base_file=target_video,
            generated_file=lipsync_video_path if lipsync_result.get('success') else None,
            parameters={
                'source_face': source_face_image,
                'reference_audio': reference_audio,
                'jailbreak_text': jailbreak_text
            },
            description='Deepfake + Voice Cloning attack',
            tags='cross_modal,deepfake,voice_clone'
        )

        return {
            'media_id': media_id,
            'video_path': lipsync_video_path if lipsync_result.get('success') else None,
            'audio_path': cloned_audio_path,
            'voice_result': voice_result,
            'lipsync_result': lipsync_result,
            'modality': 'video+audio+deepfake'
        }

    def create_full_multimedia_attack(self, base_image: str, base_audio: str,
                                      base_video: str, jailbreak_text: str,
                                      advanced: bool = True) -> Dict:
        """
        전체 멀티미디어 복합 공격

        이미지 + 오디오 + 비디오 통합 공격

        Args:
            base_image: 원본 이미지
            base_audio: 원본 오디오
            base_video: 원본 비디오
            jailbreak_text: Jailbreak 프롬프트
            advanced: 고급 공격 사용 여부 (Foolbox, ART 등)

        Returns:
            통합 공격 결과
        """
        results = {}

        # 1. 이미지 공격
        print("1. Generating adversarial image...")
        if advanced and self.foolbox_attacker:
            image_result = self.create_visual_text_attack(
                base_image, jailbreak_text, image_attack='pgd'
            )
        else:
            image_result = self.create_visual_text_attack(
                base_image, jailbreak_text, image_attack='visual_jailbreak'
            )
        results['image'] = image_result

        # 2. 오디오 공격
        print("2. Generating adversarial audio...")
        audio_result = self.create_audio_text_attack(
            base_audio, jailbreak_text, audio_attack='ultrasonic_command'
        )
        results['audio'] = audio_result

        # 3. 비디오 공격
        print("3. Generating adversarial video...")
        video_output = os.path.join(self.output_dir, 'full_multimedia_video.mp4')
        from multimodal.video_adversarial import create_jailbreak_video
        video_result = create_jailbreak_video(
            base_video,
            jailbreak_text,
            video_output,
            method='invisible_text_frames'
        )

        video_media_id = self.db.insert_media(
            media_type='video',
            attack_type='full_multimedia',
            base_file=base_video,
            generated_file=video_result['output_path'],
            parameters={'jailbreak_text': jailbreak_text},
            description='Full multimedia attack (video component)',
            tags='cross_modal,multimedia'
        )

        results['video'] = {
            'media_id': video_media_id,
            'video_path': video_result['output_path']
        }

        # 4. 크로스 모달 조합 저장
        combination_id = self.db.insert_cross_modal_combination(
            image_id=image_result.get('media_id'),
            audio_id=audio_result.get('media_id'),
            video_id=results['video'].get('media_id'),
            text_prompt_id=None,
            combination_type='full_multimedia',
            description=f'Full multimedia attack: {jailbreak_text[:50]}'
        )

        results['combination_id'] = combination_id
        results['jailbreak_text'] = jailbreak_text
        results['success'] = True

        return results

    def analyze_cross_modal_effectiveness(self, combination_id: int) -> Dict:
        """
        크로스 모달 공격 효과 분석

        Args:
            combination_id: 조합 ID

        Returns:
            분석 결과
        """
        # DB에서 조합 정보 가져오기
        # 실제 테스트 결과 분석
        # 성공률, 심각도 등 분석

        return {
            'combination_id': combination_id,
            'analysis': 'Cross-modal analysis not yet implemented',
            'recommendations': []
        }


# ========================================
# 데이터베이스 확장 메서드 추가
# ========================================

def insert_cross_modal_combination(db: ArsenalDB, image_id: Optional[int] = None,
                                  audio_id: Optional[int] = None,
                                  video_id: Optional[int] = None,
                                  text_prompt_id: Optional[int] = None,
                                  combination_type: str = 'custom',
                                  description: str = '') -> int:
    """
    크로스 모달 조합 저장 (ArsenalDB 확장 메서드)

    Args:
        db: ArsenalDB 인스턴스
        image_id: 이미지 미디어 ID
        audio_id: 오디오 미디어 ID
        video_id: 비디오 미디어 ID
        text_prompt_id: 텍스트 프롬프트 ID
        combination_type: 조합 유형
        description: 설명

    Returns:
        조합 ID
    """
    import sqlite3

    conn = sqlite3.connect(db.db_path)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO cross_modal_combinations
        (image_id, audio_id, video_id, text_prompt_id, combination_type, description)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (image_id, audio_id, video_id, text_prompt_id, combination_type, description))

    combination_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return combination_id


# ArsenalDB에 메서드 추가
ArsenalDB.insert_cross_modal_combination = insert_cross_modal_combination


# ========================================
# 고수준 API
# ========================================

def create_multimedia_jailbreak(db: ArsenalDB,
                               base_image: str,
                               base_audio: str,
                               base_video: str,
                               jailbreak_text: str) -> Dict:
    """
    간편한 멀티미디어 Jailbreak 공격 생성

    Args:
        db: ArsenalDB 인스턴스
        base_image: 원본 이미지
        base_audio: 원본 오디오
        base_video: 원본 비디오
        jailbreak_text: Jailbreak 프롬프트

    Returns:
        통합 공격 결과
    """
    attacker = CrossModalAttacker(db)
    return attacker.create_full_multimedia_attack(
        base_image, base_audio, base_video, jailbreak_text
    )


if __name__ == "__main__":
    # 테스트
    from core.database import ArsenalDB

    print("Cross-Modal Attack System")
    print("=" * 50)

    db = ArsenalDB()
    attacker = CrossModalAttacker(db)

    print("Available attack types:")
    print("  1. Visual + Text (Image Jailbreak)")
    print("  2. Audio + Text (Ultrasonic Commands)")
    print("  3. Deepfake + Voice Cloning")
    print("  4. Full Multimedia (All modalities)")

    print("\nCross-Modal Attack System ready!")
