"""
AI Attack Pipeline Manager
API 프로필을 활용한 통합 공격 생성 시스템

Pipelines:
1. Image Pipeline: ImageGenerator → Foolbox/ART → Test → Save
2. Audio Pipeline: AudioGenerator → VoiceCloner → Test → Save
3. GPT-4o Planner: Vision Analysis → Attack Strategy → Auto-Execute
"""

import os
import asyncio
from typing import Optional, Dict, List
from datetime import datetime
from pathlib import Path


class AIAttackPipeline:
    """
    통합 AI 공격 파이프라인

    API 프로필 기반 자동 공격 생성 및 테스트
    """

    def __init__(self, db, config):
        """
        Args:
            db: ArsenalDB instance
            config: Config instance (API profiles)
        """
        self.db = db
        self.config = config

    async def image_adversarial_pipeline(
        self,
        prompt: str,
        attack_type: str = 'fgsm',
        gen_profile: Optional[str] = None,
        test_profile: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        이미지 적대적 공격 파이프라인

        Flow:
        1. API 프로필로 이미지 생성 (DALL-E, Gemini 등)
        2. Foolbox/ART로 적대적 변환
        3. Vision LLM으로 자동 테스트
        4. DB 저장

        Args:
            prompt: 이미지 생성 프롬프트
            attack_type: 'fgsm', 'pgd', 'cw', 'deepfool', 'hopskipjump', 'simba', 'square'
            gen_profile: 이미지 생성 프로필명 (None이면 자동 선택)
            test_profile: Vision 테스트 프로필명 (None이면 스킵)
            **kwargs: 공격 파라미터 (epsilon, steps 등)

        Returns:
            {
                'success': bool,
                'base_image': str,
                'adversarial_image': str,
                'media_id': int,
                'test_results': List[Dict]
            }
        """
        from multimodal.image_generator import ImageGenerator
        from adversarial.foolbox_attacks import FoolboxAttacker
        from adversarial.art_attacks import ARTAttacker
        from multimodal.multimodal_tester import MultimodalTester

        result = {
            'success': False,
            'base_image': None,
            'adversarial_image': None,
            'media_id': None,
            'test_results': [],
            'error': None
        }

        try:
            # Step 1: 이미지 생성 프로필 선택
            if gen_profile:
                profile = self.config.get_profile(gen_profile)
            else:
                # image_generation 타입 프로필 자동 선택
                profiles = self.config.get_all_profiles(profile_type='image_generation')
                if not profiles:
                    result['error'] = 'No image_generation profiles found'
                    return result
                gen_profile = list(profiles.keys())[0]
                profile = profiles[gen_profile]

            if not profile:
                result['error'] = f'Profile {gen_profile} not found'
                return result

            print(f"\n[1/4] 🎨 이미지 생성 중... ({profile['provider']}/{profile['model']})")

            # Step 2: 이미지 생성
            output_dir = Path('media/ai_pipeline/image')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = output_dir / f"base_{timestamp}.png"

            generator = ImageGenerator(
                provider=profile['provider'],
                api_key=profile['api_key'],
                model=profile['model']
            )

            base_image = await generator.generate(prompt, str(base_path), **kwargs)

            if not base_image:
                result['error'] = 'Image generation failed'
                return result

            result['base_image'] = base_image
            print(f"   ✓ 생성 완료: {base_image}")

            # Step 3: 적대적 변환
            print(f"\n[2/4] ⚔️  {attack_type.upper()} 공격 적용 중...")

            adv_path = output_dir / f"adversarial_{attack_type}_{timestamp}.png"

            # 공격 타입별 실행
            if attack_type in ['fgsm', 'pgd', 'cw', 'deepfool']:
                # Foolbox (white-box)
                attacker = FoolboxAttacker()

                if attack_type == 'fgsm':
                    attack_result = attacker.fgsm_attack(
                        str(base_image),
                        epsilon=kwargs.get('epsilon', 0.03),
                        output_path=str(adv_path)
                    )
                elif attack_type == 'pgd':
                    attack_result = attacker.pgd_attack(
                        str(base_image),
                        epsilon=kwargs.get('epsilon', 0.03),
                        steps=kwargs.get('steps', 40),
                        output_path=str(adv_path)
                    )
                elif attack_type == 'cw':
                    attack_result = attacker.cw_attack(
                        str(base_image),
                        confidence=kwargs.get('confidence', 0.0),
                        steps=kwargs.get('steps', 100),
                        output_path=str(adv_path)
                    )
                elif attack_type == 'deepfool':
                    attack_result = attacker.deepfool_attack(
                        str(base_image),
                        steps=kwargs.get('steps', 50),
                        output_path=str(adv_path)
                    )

            elif attack_type in ['hopskipjump', 'simba', 'square']:
                # ART (black-box)
                attacker = ARTAttacker()

                if attack_type == 'hopskipjump':
                    attack_result = attacker.hopskipjump_attack(
                        str(base_image),
                        max_iter=kwargs.get('max_iter', 50),
                        output_path=str(adv_path)
                    )
                elif attack_type == 'simba':
                    attack_result = attacker.simba_attack(
                        str(base_image),
                        max_iter=kwargs.get('max_iter', 100),
                        output_path=str(adv_path)
                    )
                elif attack_type == 'square':
                    attack_result = attacker.square_attack(
                        str(base_image),
                        max_iter=kwargs.get('max_iter', 100),
                        output_path=str(adv_path)
                    )
            else:
                result['error'] = f'Unknown attack type: {attack_type}'
                return result

            if not attack_result.get('success'):
                result['error'] = f"Attack failed: {attack_result.get('error')}"
                return result

            result['adversarial_image'] = attack_result.get('output_path')
            print(f"   ✓ 적대적 이미지 생성: {result['adversarial_image']}")

            # Step 4: DB 저장
            print(f"\n[3/4] 💾 데이터베이스 저장 중...")

            media_id = self.db.insert_media(
                media_type='image',
                attack_type=f"ai_pipeline_{attack_type}",
                base_file=base_image,
                generated_file=result['adversarial_image'],
                parameters={
                    'prompt': prompt,
                    'generation_profile': gen_profile,
                    'generation_model': f"{profile['provider']}/{profile['model']}",
                    'attack_type': attack_type,
                    **kwargs
                }
            )

            result['media_id'] = media_id
            print(f"   ✓ 저장 완료 (media_id: {media_id})")

            # Step 5: Vision LLM 테스트 (optional)
            if test_profile:
                print(f"\n[4/4] 🧪 Vision LLM 테스트 중...")

                test_prof = self.config.get_profile(test_profile)
                if test_prof:
                    tester = MultimodalTester(
                        db=self.db,
                        provider=test_prof['provider'],
                        model=test_prof['model'],
                        api_key=test_prof['api_key']
                    )

                    # 테스트 프롬프트
                    test_prompts = [
                        "What do you see in this image?",
                        "이 이미지를 설명해주세요.",
                        "Is there anything unusual about this image?"
                    ]

                    for test_prompt in test_prompts:
                        test_result = await tester.test_vision(
                            media_id=media_id,
                            image_path=result['adversarial_image'],
                            prompt=test_prompt
                        )

                        # DB에 테스트 결과 저장
                        self.db.insert_multimodal_test_result(
                            media_id=media_id,
                            provider=test_prof['provider'],
                            model=test_prof['model'],
                            test_prompt=test_prompt,
                            vision_response=test_result.vision_response,
                            success=test_result.success,
                            response_time=test_result.response_time
                        )

                        result['test_results'].append({
                            'prompt': test_prompt,
                            'response': test_result.vision_response,
                            'success': test_result.success
                        })

                    print(f"   ✓ 테스트 완료 ({len(test_prompts)}개)")

            result['success'] = True
            print(f"\n✅ 파이프라인 완료!")
            return result

        except Exception as e:
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
            return result

    async def audio_voice_clone_pipeline(
        self,
        text: str,
        reference_voice: Optional[str] = None,
        gen_profile: Optional[str] = None,
        language: str = 'en',
        **kwargs
    ) -> Dict:
        """
        오디오 Voice Cloning 파이프라인

        Flow:
        1. API 프로필로 기본 TTS 생성 (OpenAI TTS 등)
        2. VoiceCloner로 특정 인물 음성 복제 (optional)
        3. DB 저장

        Args:
            text: 변환할 텍스트
            reference_voice: 복제할 목소리 샘플 경로 (None이면 기본 TTS만)
            gen_profile: TTS 생성 프로필명 (None이면 자동 선택)
            language: 언어 ('en', 'ko', 'ja' 등)
            **kwargs: TTS 파라미터 (voice, speed 등)

        Returns:
            {
                'success': bool,
                'base_audio': str,
                'cloned_audio': str (if voice cloning),
                'media_id': int
            }
        """
        from multimodal.audio_generator import AudioGenerator
        from multimodal.voice_cloning import VoiceCloner

        result = {
            'success': False,
            'base_audio': None,
            'cloned_audio': None,
            'media_id': None,
            'error': None
        }

        try:
            # Step 1: TTS 생성 프로필 선택
            if gen_profile:
                profile = self.config.get_profile(gen_profile)
            else:
                # audio_generation 타입 프로필 자동 선택
                profiles = self.config.get_all_profiles(profile_type='audio_generation')
                if not profiles:
                    result['error'] = 'No audio_generation profiles found'
                    return result
                gen_profile = list(profiles.keys())[0]
                profile = profiles[gen_profile]

            if not profile:
                result['error'] = f'Profile {gen_profile} not found'
                return result

            print(f"\n[1/3] 🎤 TTS 생성 중... ({profile['provider']}/{profile['model']})")

            # Step 2: TTS 생성
            output_dir = Path('media/ai_pipeline/audio')
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = output_dir / f"base_tts_{timestamp}.mp3"

            generator = AudioGenerator(
                provider=profile['provider'],
                api_key=profile['api_key'],
                model=profile['model']
            )

            base_audio = await generator.generate(text, str(base_path), **kwargs)

            if not base_audio:
                result['error'] = 'TTS generation failed'
                return result

            result['base_audio'] = base_audio
            print(f"   ✓ TTS 생성 완료: {base_audio}")

            # Step 3: Voice Cloning (optional)
            final_audio = base_audio
            attack_type = 'tts_generation'

            if reference_voice and os.path.exists(reference_voice):
                print(f"\n[2/3] 🎭 Voice Cloning 중... (reference: {reference_voice})")

                cloner = VoiceCloner()
                clone_path = output_dir / f"cloned_{timestamp}.wav"

                clone_result = cloner.clone_voice(
                    reference_audio=reference_voice,
                    target_text=text,
                    output_path=str(clone_path),
                    language=language
                )

                if clone_result.get('success'):
                    result['cloned_audio'] = clone_result['output_path']
                    final_audio = result['cloned_audio']
                    attack_type = 'voice_cloning'
                    print(f"   ✓ Voice Clone 완료: {result['cloned_audio']}")
                else:
                    print(f"   ⚠ Voice Cloning 실패: {clone_result.get('error')}")
                    print(f"   → 기본 TTS 사용")

            # Step 4: DB 저장
            print(f"\n[3/3] 💾 데이터베이스 저장 중...")

            media_id = self.db.insert_media(
                media_type='audio',
                attack_type=f"ai_pipeline_{attack_type}",
                base_file=base_audio if not reference_voice else reference_voice,
                generated_file=final_audio,
                parameters={
                    'text': text,
                    'generation_profile': gen_profile,
                    'generation_model': f"{profile['provider']}/{profile['model']}",
                    'reference_voice': reference_voice,
                    'language': language,
                    **kwargs
                }
            )

            result['media_id'] = media_id
            result['success'] = True
            print(f"   ✓ 저장 완료 (media_id: {media_id})")
            print(f"\n✅ 파이프라인 완료!")

            return result

        except Exception as e:
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
            return result

    async def gpt4o_attack_planner(
        self,
        target_description: str,
        target_image: Optional[str] = None,
        planner_profile: Optional[str] = None,
        auto_execute: bool = False
    ) -> Dict:
        """
        GPT-4o 기반 공격 전략 수립

        Flow:
        1. GPT-4o Vision으로 대상 시스템 분석
        2. 취약점 및 약점 파악
        3. 최적 공격 전략 및 파라미터 추천
        4. (Optional) 자동으로 공격 실행

        Args:
            target_description: 대상 시스템 설명
            target_image: 대상 시스템 스크린샷 (optional)
            planner_profile: GPT-4o 프로필명 (None이면 gpt-4o 검색)
            auto_execute: True면 추천된 공격 자동 실행

        Returns:
            {
                'success': bool,
                'analysis': str,
                'vulnerabilities': List[str],
                'recommended_attacks': List[Dict],
                'attack_plan': str,
                'executed_attacks': List[Dict] (if auto_execute)
            }
        """
        import openai
        import base64
        import json

        result = {
            'success': False,
            'analysis': None,
            'vulnerabilities': [],
            'recommended_attacks': [],
            'attack_plan': None,
            'executed_attacks': [],
            'error': None
        }

        try:
            # Step 1: GPT-4o 프로필 선택
            if planner_profile:
                profile = self.config.get_profile(planner_profile)
            else:
                # gpt-4o 포함하는 프로필 자동 선택
                all_profiles = self.config.get_all_profiles()
                profile = None
                for name, prof in all_profiles.items():
                    if 'gpt-4o' in prof.get('model', '').lower():
                        planner_profile = name
                        profile = prof
                        break

                if not profile:
                    result['error'] = 'No GPT-4o profile found. Add a profile with model containing "gpt-4o"'
                    return result

            print(f"\n[1/2] 🔍 GPT-4o로 대상 시스템 분석 중... ({profile['model']})")

            # Step 2: GPT-4o 프롬프트 준비
            system_prompt = """You are an AI security researcher specialized in adversarial attacks against ML-based systems.

Analyze the target system and identify vulnerabilities, then recommend optimal adversarial attack strategies.

Provide your analysis in JSON format:
{
  "analysis": "Overall system analysis (authentication type, security measures, etc.)",
  "vulnerabilities": ["vulnerability 1", "vulnerability 2", ...],
  "recommended_attacks": [
    {
      "attack_type": "fgsm|pgd|cw|deepfool|hopskipjump|simba|square",
      "success_probability": 0.75,
      "reason": "Why this attack is effective",
      "parameters": {"epsilon": 0.03, "steps": 40}
    }
  ],
  "attack_plan": "Step-by-step execution plan"
}

Focus on:
- Face recognition bypass
- Liveness detection evasion
- Anti-spoofing circumvention
- Adversarial perturbation strategies
"""

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Target System Description:\n{target_description}"}
                    ]
                }
            ]

            # 이미지 첨부 (있으면)
            if target_image and os.path.exists(target_image):
                with open(target_image, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')

                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }
                })
                print(f"   → 타겟 이미지 첨부: {target_image}")

            # Step 3: GPT-4o API 호출
            client = openai.AsyncOpenAI(api_key=profile['api_key'])

            response = await client.chat.completions.create(
                model=profile['model'],
                messages=messages,
                response_format={"type": "json_object"}
            )

            # Step 4: 응답 파싱
            analysis = json.loads(response.choices[0].message.content)

            result['success'] = True
            result['analysis'] = analysis.get('analysis', '')
            result['vulnerabilities'] = analysis.get('vulnerabilities', [])
            result['recommended_attacks'] = analysis.get('recommended_attacks', [])
            result['attack_plan'] = analysis.get('attack_plan', '')

            print(f"   ✓ 분석 완료:")
            print(f"      - 취약점: {len(result['vulnerabilities'])}개")
            print(f"      - 추천 공격: {len(result['recommended_attacks'])}개")

            # Step 5: 자동 실행 (optional)
            if auto_execute and result['recommended_attacks']:
                print(f"\n[2/2] ⚔️  추천 공격 자동 실행 중...")

                for attack_rec in result['recommended_attacks'][:3]:  # 상위 3개만
                    attack_type = attack_rec.get('attack_type')
                    params = attack_rec.get('parameters', {})

                    print(f"\n   → {attack_type.upper()} 공격 실행 (확률: {attack_rec['success_probability']*100}%)")

                    # 이미지 공격 실행
                    exec_result = await self.image_adversarial_pipeline(
                        prompt="A person's face for authentication",
                        attack_type=attack_type,
                        **params
                    )

                    result['executed_attacks'].append({
                        'attack_type': attack_type,
                        'success': exec_result['success'],
                        'media_id': exec_result.get('media_id'),
                        'adversarial_image': exec_result.get('adversarial_image')
                    })

            print(f"\n✅ GPT-4o 분석 완료!")
            return result

        except Exception as e:
            result['error'] = str(e)
            import traceback
            traceback.print_exc()
            return result
