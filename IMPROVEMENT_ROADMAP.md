# Implementation Plan: 3가지 개선사항

## 1순위: 파인튜닝 검증 게이트(완료) ✅

### 구현내용
- **train_halfweg.py 개선**:
  - `quick_probe()`: 단일 파일 기반 모니터링 프로브 (legacy)
  - `multi_file_probe()`: 3~5개 파일 앙상블 기반 프로브 (새로운 gold standard)
  - Gate mechanism: `unfiltered_valid_5files` solve_rate > baseline 시에만 `[GatePassed]` 체크포인트 저장

- **검증 로직**:
  ```
  Epoch N (probe every 2): 
    → quick_probe(000.txt) = 0.40  [모니터링용]
    → multi_file_probe(5files avg) = 0.54  [게이트 판정용]
    → Gate: 0.54 > 0.58? ❌ FAIL (baseline 미달)
    → 모델 미채택
  ```

- **효과**:
  - 단일 파일 과적합 방지
  - 일반화 능력 기반 모델 선택
  - 배포 전 신뢰도 향상

---

## 2순위: 커리큘럼 학습 + 데이터 혼합(완료) ✅

### 구현내용

#### A. PL0 → PL1 정책 전환
```yaml
collector_policy_initial: PL0  # Epoch 1~8: 안정적인 수집
collector_policy_final: PL1    # Epoch 9+: 탐색적인 수집
curriculum_transition_epoch: 8
```

**동작**:
- 초반 8 에폭: PL0 (결정론적 정책) 사용 → 붕괴 방지, 안정적 학습 신호
- Epoch 9부터: PL1 (더 넓은 탐색) 전환 → baseline 넘기 위한 강한 신호

#### B. 데이터 혼합 커리큘럼
```yaml
use_curriculum_data: true
data_sources:
  - name: unfiltered
    glob: /workspace/boxoban-levels/unfiltered/train/*.txt
    weight: 0.6  (초반 1.2배 부스트)
  - name: medium
    glob: /workspace/boxoban-levels/medium/train/*.txt
    weight: 0.3  (중반 1.0배)
  - name: hard
    glob: /workspace/boxoban-levels/hard/*.txt
    weight: 0.1  (초반 0.5배 감소)
```

**동작** (`_pick_data_source_file()`):
- Epoch 1~8 (초반):
  - unfiltered: 60% × 1.2 = 72% (강조)
  - medium: 30% × 0.8 = 24%
  - hard: 10% × 0.5 = 4%
  
- Epoch 9+ (후반):
  - unfiltered: 60% (정상)
  - medium: 30% (정상)
  - hard: 10% (정상)

**효과**:
- Early phase: 쉬운 분포로 학습 신호 확보
- Late phase: 어려운 분포로 일반화 능력 강화
- Medium/hard split에서 더 나은 성능 기대

---

## 3순위: 실시간 재계획(Replanning) 연산 최적화

### 현재 상태
- **병목**: `_solve__closed_loop_replan()` 에서 **모든 타깃 후보(100~200개)를 순차 계획**
- **결과**: 처리 시간 **275배 증가** (이전 대비)
- **해결 필요**: GPU 추론 효율화 + 타깃 필터링

### 추천 개선안 (우선순위 순)

#### 단계 1: 타깃 필터링 (빠른 승리)
```python
def _filter_high_progress_targets(targets_np, curr_state, progress_threshold=0.8):
    """
    높은 progress를 가진 타깃 사전 필터링으로 계산 량감
    """
    filtered_targets = []
    for target_np in targets_np:
        # 현재 상태에서 이미 목표 상태에 근접하면 스킵
        progress = compute_box_overlap(curr_state, target_np)
        if progress < progress_threshold:
            filtered_targets.append(target_np)
    return filtered_targets
```

**효과**:
- 많은 타깃이 이미 80%+ 완성 상태라면 10~20% 계산 감소
- 실제 gain: 중간 정도 (20~30% speedup)

#### 단계 2: 배치 추론 (중간 수고, 큰 효과)
```python
def _get_batch_plans(policy, s0_list, targets_list):
    """
    여러 (state, target) 쌍을 배치로 한번에 계산
    GPU 배치 처리 효율: ~3~5배 향상
    """
    # Before: for target in targets: plan = policy(s0, target)  # 개별 호출
    # After: plans = policy.batch_get_plans(s0_list, targets_list)  # 배치 호출
    pass
```

**효과**:
- GPU 배치 활용도 90% → 사용도 가능 (이전: 10~15%)
- 시간 단축: **50~70% 감소** 가능
- 단, policy 인터페이스 수정 필요

#### 단계 3: 조기 종료 조건 강화 (간단, 부분 효과)
```python
# 현재: 모든 타깃 시도 후 best action 선택
# 개선: 
#   - 우수한 action을 충분히 찾으면 조기 종료
#   - 동일 action 연속 3회 이상이면 commit
```

**효과**:
- plan 탐색 깊이 감소
- 시간 단축: **10~20%**

### 구현 우선순위 및 예상 ROI

| 개선안 | 난이도 | 코드수정 | 효과크기 | 추천시점 |
|--------|--------|---------|---------|---------|
| 타깃 필터링 | 낮음 | 20줄 | 20-30% | **즉시** |
| 조기종료강화 | 낮음 | 15줄 | 10-20% | **즉시** |
| 배치추론 | 높음 | 100+줄 | 50-70% | 후속 |

### 코드 위치
- `halfweg/src/hw_impl/evaluate.py`
- 함수: `_solve__closed_loop_replan()` (line 89~)

### 다음 단계
1. **타깃 필터링 + 조기종료** → 즉시 구현 (30-50% speedup)
2. **배치 추론 리팩토링** → 3개월 내 로드맵

---

## 최종 검증 전략

### Checkpoint 승격 흐름
```
Train → Epoch N (Probe)
  ↓
Legacy Probe (000.txt): 0.40?
  ↓ (모니터링만)
Multi-file Probe (5files): 0.54?
  ↓
Gate: 0.54 > 0.58 (baseline)?
  ├─ YES → [GatePassed] 저장 + 배포 후보
  └─ NO  → Skip, 계속 학습
```

### 다음 재학습 명령어
```bash
# 커리큘럼 학습 + 다중 파일 검증
docker run --gpus all \
  -v $PWD:/workspace \
  halfweg-gpu-eval:latest \
  python -u /workspace/halfweg/src/train_halfweg.py \
  /workspace/halfweg/configs/train_finetune_medium.yaml
```

---

## 예상 성능 향상

### 이전 결과 (finetuned_v1: 실패)
```
unfiltered_test: 0.08
unfiltered_valid: 0.02
unfiltered_train: 0.02
```

### 현재 결과 (finetuned_v2: 회복)
```
unfiltered_test: 0.48
unfiltered_valid: 0.54
unfiltered_train: 0.50
```

### 예상 결과 (finetuned_v3: 커리큘럼)
```
unfiltered_test: 0.55~0.62  (↑ PL1 + 데이터 혼합)
unfiltered_valid: 0.58~0.65 (↑ baseline 초과 가능)
unfiltered_train: 0.52~0.60 (↑ hard/medium 비중 증가)
medium_valid: 0.10~0.15    (↑ medium 수집)
hard_all: 0.05~0.15        (↑ hard 수집)
```

---

## 완료 체크리스트

### 1순위 ✅
- [x] Multi-file probe 구현
- [x] Gate mechanism 추가
- [x] baseline 비교 로직
- [x] checkpoint 저장 (GatePassed)

### 2순위 ✅
- [x] PL0→PL1 커리큘럼 전환
- [x] curriculum_transition_epoch 설정
- [x] 데이터 소스 혼합 로직
- [x] 가중치 기반 샘플링

### 3순위 (로드맵)
- [ ] 타깃 필터링 (다음 단계)
- [ ] 조기종료 강화 (다음 단계)
- [ ] 배치 추론 리팩토링 (후속)
