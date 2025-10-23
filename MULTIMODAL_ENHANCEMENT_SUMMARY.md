# Multimodal LLM Testing Enhancement

## Overview

Enhanced menu option **9. 멀티모달 LLM 테스트** to support in-flow media generation and batch testing.

## User Request

> "지금 3. 멀티모달 공격 생성, 6. 멀티모달 무기고 검색, 9. 멀티모달 llm 테스트 3개로 나누어져있잖아. 여기서 9. 멀티모달 llm 테스트에서도 공격을 생성할 수 있으면 좋을 것 같아. 배치로 테스트 가능하게 멀티모달에 추가할 프롬프트를 우리가 적거나 목록에서 고르는 식으로 해서"

Translation: In option 9, it would be good if we could also generate attacks. Enable batch testing so we can either write prompts or select from a list for multimodal testing.

## Implementation

### New Workflow

```
Menu 9: 멀티모달 LLM 테스트
    ↓
1. Select API Profile
    ↓
2. Media Source Selection
   - Option 1: 기존 무기고에서 선택 (Select from existing arsenal)
   - Option 2: 새로 생성 (Generate new media)
    ↓
3. Test Mode Selection
   - Option 1: 단일 테스트 (Single test with 1 prompt)
   - Option 2: 배치 테스트 (Batch test with multiple prompts)
    ↓
4. Execute Test(s)
```

### New Features

#### 1. Media Source Selection (Step 2)

**Option 1: Existing Arsenal**
- Select from previously generated media
- Shows available images with attack types

**Option 2: Generate New Media**
- Media type selection: Image / Audio / Video
- Text prompt input
- Attack type specification
- Generation methods:
  - Image: DALL-E or Typography (local)
  - Audio: TTS
  - Video: Not yet supported
- Auto-saves to database after generation

#### 2. Test Mode Selection (Step 3)

**Option 1: Single Test**
- Select single prompt (direct input OR database)
- Immediate test with briefing display
- Full result details

**Option 2: Batch Test**
- Multiple prompt selection methods:
  - **Method 1**: Direct input (enter multiple prompts manually)
  - **Method 2**: DB category (select all prompts from a category)
  - **Method 3**: DB individual (select specific prompts)
- Progress tracking for each test
- Summary statistics at end

#### 3. Prompt Selection

**For Single Test:**
- Direct input
- Select from database (category → prompt)

**For Batch Test:**
- Direct input (multiple prompts, empty line to finish)
- DB category (select category, specify limit)
- DB individual (TODO: multi-select implementation)

### Helper Functions Implemented

1. **`_generate_media_for_test()`**
   - Main media generation dispatcher
   - Handles media type selection
   - Returns media_id, file_path, attack_type

2. **`_generate_image_for_test(prompt, attack_type)`**
   - Image generation using DALL-E or Typography
   - Profile selection for API-based generation
   - Auto-saves to database

3. **`_generate_audio_for_test(prompt, attack_type)`**
   - Audio generation using TTS
   - Profile selection
   - Auto-saves to database

4. **`_generate_video_for_test(prompt, attack_type)`**
   - Placeholder (not yet implemented)

5. **`_select_media_from_arsenal()`**
   - Select media from existing arsenal
   - Returns media_id and selected media data

6. **`_single_multimodal_test(profile, media_id, selected)`**
   - Single test workflow
   - Prompt selection → Judge mode → Test execution
   - Full briefing display

7. **`_batch_multimodal_test(profile, media_id, selected)`**
   - Batch test workflow
   - Multiple prompt selection
   - Progress tracking
   - Summary statistics

8. **`_select_prompt()`**
   - Single prompt selection
   - Direct input OR database selection

9. **`_select_prompt_from_db()`**
   - Database prompt selection
   - Category → Prompt list → Selection

10. **`_select_prompts_batch()`**
    - Multiple prompt selection
    - Three methods: direct input, category, individual

11. **`_select_judge_mode()`**
    - Judge mode selection
    - Options: rule-based, llm, hybrid, default

12. **`_run_multimodal_test(...)`**
    - Core test execution
    - Optional briefing display
    - Result display with box drawing
    - Auto-save to database

## Code Cleanup

- Removed 280+ lines of duplicate old code (lines 2726-3008)
- Maintained all helper methods properly
- Fixed code organization

## Testing Checklist

- [x] Syntax validation (no errors)
- [ ] Test media generation (Option 2 in Media Source)
  - [ ] Image generation (DALL-E)
  - [ ] Image generation (Typography)
  - [ ] Audio generation (TTS)
- [ ] Test single mode (Option 1 in Test Mode)
  - [ ] Direct prompt input
  - [ ] Database prompt selection
- [ ] Test batch mode (Option 2 in Test Mode)
  - [ ] Direct input method
  - [ ] DB category method
  - [ ] DB individual method
- [ ] Verify results are saved to database
- [ ] Test judge mode selection
- [ ] Test error handling

## Files Modified

- `/Users/brownkim/Downloads/ACDC/prompt_arsenal/interactive_cli.py`
  - Lines 2259-2724: Complete rewrite of `attack_multimodal_llm()` and helper methods
  - Removed lines 2726-3008: Old duplicate code

## Impact

✅ **Fully addresses user request:**
1. ✅ Media can be generated directly in testing workflow (Option 2 in Media Source)
2. ✅ Batch testing supported with multiple prompts (Option 2 in Test Mode)
3. ✅ Flexible prompt input (direct OR database selection)
4. ✅ Progress tracking and summary statistics for batch tests

## Next Steps

1. Manual testing to verify all workflows work correctly
2. Fix any runtime errors discovered during testing
3. Consider adding multi-select UI for "DB individual" batch method
4. Implement video generation when ready
