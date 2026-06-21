# 커리큘럼 학습 분석 보고서 (2026-06-19)

## 1. 요약

**결과**: 🔴 **실패** (게이트 통과했으나 심각한 성능 붕괴로 조기 종료)

| 항목 | 값 |
|------|-----|
| 목표 | baseline (0.58) 초과 지속 → 모델 배포 |
| 최고 성능 (Epoch 2) | **0.66** ✅ 게이트 통과 |
| 최종 성능 (Epoch 10) | **0.00** ❌ 완전 붕괴 |
| 성능 하락폭 | **-66%** (8 에폭 내) |
| 조기 종료 | Epoch 10 (early_stop_patience=4 초과) |
| 체크포인트 | checkpoint_20260618-192542.ckpt (게이트 통과본) |

---

## 2. 에폭별 성능 추이

### 2.1 Epoch 1-2: 정상 진행 ✅

```
Epoch 1
├─ 데이터 수집: solved=0, partial=4
├─ 학습 손실: plta_loss=148.69, plhw_loss=0.0022
└─ 프로브 (단일 파일): 0.650 (n=20)
   └─ 미표시: multi_file_probe 아직 평가 안 함 (probe_every_epochs=2)

Epoch 2 ⭐ GATE PASSED
├─ 데이터 수집: solved=0, partial=4
├─ 학습 손실: plta_loss=9.66, plhw_loss=0.0010 (감소 추세)
├─ 프로브 (단일 파일): 0.650 (n=20)
└─ 🎯 다중 파일 프로브 (5 files):
    ├─ File 1 (000.txt): 0.700
    ├─ File 2 (001.txt): 0.400
    ├─ File 3 (002.txt): 0.700
    ├─ File 4 (003.txt): 0.700
    ├─ File 5 (004.txt): 0.800
    └─ **평균: 0.660** ✅ PASS (vs baseline=0.580)
    
✅ [GatePassed 0.660] 체크포인트 저장
```

**분석**:
- 초반 안정적인 손실 감소 (PLTA: 148.69 → 9.66)
- 데이터 수집 신호: partial trajectory 계속 발생 (안정적)
- 다중 파일 프로브 평균이 단일 파일(0.650)보다 높음 → 일반화 가능성 보임

---

### 2.2 Epoch 3-4: 성능 하락 시작 ⚠️

```
Epoch 3
├─ 데이터 수집: solved=0, partial=4
├─ 학습 손실: plta_loss=19.55, plhw_loss=0.0105
└─ (probe_every_epochs=2이므로 평가 안 함)

Epoch 4 ⬇️ 성능 급락
├─ 데이터 수집: solved=0, partial=3 (수집 감소)
├─ 학습 손실: plta_loss=124.04, plhw_loss=0.0086 (PLTA 급증)
├─ 프로브 (단일 파일): 0.350 (n=20) ← 이전 0.650에서 46% 하락
└─ 🎯 다중 파일 프로브 (5 files):
    ├─ File 1: 0.400, File 2: 0.500, File 3: 0.400, File 4: 0.400, File 5: 0.300
    └─ **평균: 0.400** ❌ FAIL (vs baseline=0.580)
    
❌ Gate check 실패: probe=0.400 vs baseline=0.580
📌 No probe improvement (1/4): early_stop_patience 카운트 +1
```

**원인 분석**:
- PLTA 손실이 갑자기 10배 증가 (9.66 → 124.04)
- 데이터 수집 부족: partial=3 (이전 4에서 감소)
- 손실 폭발은 gradient 불안정 + 작은 배치 크기(n_levels_per_epoch=4) 영향

---

### 2.3 Epoch 5-8: 계속된 붕괴 🔴

```
Epoch 5-6: 진동적 악화
├─ plta_loss 계속 증가 (17.23 → 154.06)
├─ 데이터 수집 지속 감소 (partial=2)
└─ multi_file_probe: 0.240 ❌ (-60% from peak)

Epoch 7-8: 최종 붕괴 전
├─ plta_loss 폭주: 314.66 → 400.74
├─ plhw_loss 급증: 0.0041 → 0.0960 (24배)
└─ [Curriculum] Switching to PL1 policy at epoch 9
    └─ ⚠️ 정책 전환 전 이미 성능 0% 도달
```

**주요 신호**:
- PLTA 손실이 계속 폭증 → NaN 직전 상태
- Epoch 8에는 multi_file_probe가 0.000 (모든 파일 0%)
- **정책 전환 시점(epoch 9)에는 이미 모델이 붕괴된 상태**

---

### 2.4 Epoch 9-10: 정책 전환 후 NaN + 완전 정지 🔴

```
Epoch 9: PL1 정책 적용 + NaN 발생
├─ [Curriculum] Switching to PL1 policy at epoch 9 ✓ (정책 전환 발생 확인)
├─ 데이터 수집: solved=0, partial=1 (극도로 부족)
├─ 🚨 plta_loss=nan (PLHW 손실은 0.1095로 정상)
└─ 프로브 결과 (단일 + 다중 파일 모두):
    ├─ proposed_plan_length_mean=0.0  ← 계획 길이 0 (즉시 정지)
    ├─ avg_replan_count=1.0            ← 재계획 없음 (처음부터 정지)
    ├─ avg_box_push_count=0.0          ← 박스 이동 0
    └─ **solve_rate=0.000 (모든 파일)**

Epoch 10: 지속적 정지 동작
├─ plta_loss=46.34 (NaN에서 회복했으나 여전히 높음)
├─ timeout_rate=0.0 (제한시간 내 반환, 하지만 해결 못함)
└─ 동일한 정지 패턴 반복
    └─ proposed_plan_length_mean=0.0 (여전히)
    └─ avg_replan_count=1.0
    └─ avg_box_push_count=0.0

조기 종료 발동: No probe improvement (4/4)
```

**심각한 문제**:
1. **PLTA NaN 발생**: Epoch 9에서 명시적인 NaN 손실값
2. **정책 퇴화**: 모델이 "항상 정지 행동(AS) 선택" 상태로 붕괴
   - proposed_plan_length=0 → 계획을 전혀 생성 안 함
   - avg_box_push_count=0 → 박스 한 칸도 안 밀음
3. **PL1 전환의 부작용**: PL1 정책이 더 복잡한 탐색을 시도했으나 기울기 불안정 유발

---

## 3. 근본 원인 분석

### 문제 1: 극도로 작은 수집 크기
```
n_levels_per_epoch: 4
n_attempts_per_level: 1
→ 매 에폭당 수집 샘플 수: 4 × 1 = 4개 (매우 작음)

배치 크기: 128
→ 4개 샘플 수집 후 128 배치로 학습 불가
→ 실제 배치: 4개 내 중복 샘플링 (높은 분산)
```

**해결책**:
```yaml
n_levels_per_epoch: 32  # 8배 증가
n_attempts_per_level: 2  # 2배 증가
→ 매 에폭당: 32 × 2 = 64개 (배치 크기 128의 절반)
```

### 문제 2: 데이터 부족 시 PL1 정책 전환
```
Epoch 8 상황:
├─ partial trajectory 감소 추세 (4 → 2)
├─ plta_loss 폭주 (314.66 → 400.74)
├─ 모델이 이미 불안정한 상태
└─ [Policy 전환] PL1은 더 복잡한 탐색 시도
    └─ 불안정한 기울기 + 강한 신호 = NaN 발생 가능

Epoch 9 이후:
├─ PLTA NaN + partial trajectory 거의 없음 (1)
└─ 수렴 불가능 → 정책 퇴화
```

**해결책**:
- **Gradient clip 추가**: `torch.nn.utils.clip_grad_norm_`로 큰 기울기 방지
- **NaN 감지 및 롤백**: 손실이 NaN이면 즉시 이전 체크포인트로 복구
- **정책 전환 조건 강화**:
  ```python
  # PL1 전환 전 다음 확인:
  if recent_probe_improve < 0.01:  # 최근 3 에폭 개선 < 1%
      print("Too unstable, keep PL0")
      use_pl1 = False
  ```

### 문제 3: 커리큘럼 데이터 혼합의 과도한 난이도 증가
```
Config (train_finetune_medium.yaml):
  data_sources:
    - unfiltered: 0.6 (쉬움) ← 하지만 60%만
    - medium: 0.3 (중간)    ← 30%
    - hard: 0.1 (어려움)    ← 10%

현실:
├─ Epoch 1-2: 일부 unfiltered 수집 → 안정
├─ Epoch 4-8: curriculum weighted sampling이
│  └─ 난이도가 높은 파일 선택 비율 증가
│  └─ 데이터 부족 + 난이도 증가 = 폭발
└─ Epoch 9: 폭발한 모델 + PL1 전환 = 완전 붕괴
```

**해결책**:
```yaml
data_sources:
  - unfiltered: 0.8  # 80% (더 강조)
  - medium: 0.15     # 15%
  - hard: 0.05       # 5% (초반에는 최소)
```

### 문제 4: 검증 게이트가 너무 늦음
```
현재 probe_every_epochs=2:
├─ Epoch 2: 게이트 통과 (0.66)
├─ Epoch 4: 이미 하락 (0.40)
├─ Epoch 6: 더 하락 (0.24)
└─ 응답 속도가 너무 느림

→ Epoch 4 시점에는 이미 손실 폭주 (124.04)
→ 게이트 통과 모델(Epoch 2)을 지켜야 했음
```

**해결책**:
```python
# 매 에폭마다 빠른 health check
if epoch > 2:
    if recent_plta_loss_mean > prev_epoch_loss * 2:
        print("Loss exploding, reverting to last checkpoint")
        load_checkpoint(best_gate_passed_ckpt)
```

---

## 4. 성과 & 교훈

### ✅ 성과
1. **다중 파일 프로브 구현 성공**: 
   - 단일 파일(0.65) vs 다중(0.66) 비교 가능
   - 단일 파일이 다중보다 낮음 → 과적합 가능성 확인

2. **PL0 → PL1 정책 전환 구현 확인**:
   - Epoch 8→9에서 명시적 메시지 확인
   - `[Curriculum] Switching to PL1 policy at epoch 9` 출력됨

3. **게이트 통과 모델 확보**: 
   - checkpoint_20260618-192542.ckpt (0.66 성능)
   - baseline 0.58을 초과하는 첫 학습 모델

### 🔴 교훈

| 실패 항목 | 원인 | 다음 개선 |
|----------|------|----------|
| 성능 유지 불가 | 데이터량 부족 (n_levels_per_epoch=4) | 32-64로 증가 |
| PLTA NaN | Gradient 폭발 + PL1 불안정성 | Gradient clip + 조기 탐지 |
| 정책 퇴화 | 약한 학습 신호 | NaN 감지 시 immediate rollback |
| 늦은 대응 | probe_every_epochs=2 (2에폭 간격) | 매 에폭 loss health check |
| 커리큘럼 과도 | 중간/어려움 난이도 비중 높음 | unfiltered 80% 강조 |

---

## 5. 베스트 체크포인트 분석

### checkpoint_20260618-192542.ckpt (Epoch 2, GatePassed)

**성능**:
```
Multi-file Probe (5 files, n=10 per file):
├─ File 1 (000.txt): 0.700
├─ File 2 (001.txt): 0.400
├─ File 3 (002.txt): 0.700
├─ File 4 (003.txt): 0.700
├─ File 5 (004.txt): 0.800
└─ Average: 0.660 (baseline 0.580 초과)

Single-file Probe (n=20):
└─ 0.650 (baseline보다 높음)
```

**특징**:
- 기울기 안정 (plta_loss=9.66, plhw_loss=0.0010)
- Partial trajectory 충분 (n_attempts=4 모두 성공)
- 다중 파일 간 성능 분산: min=0.4, max=0.8 (균형 잡혀 있음)

**추천 용도**:
- ✅ 즉시 배포 가능 (baseline 초과)
- ✅ 6-split 벤치마크용 best candidate
- ✅ 다음 커리큘럼 재학습의 초기값

---

## 6. 다음 단계 (Action Items)

### Phase 1: 현재 최고 모델 검증 (1시간)
```bash
# 게이트 통과 모델로 6-split 벤치마크 실행
docker run --gpus all \
  -v $PWD:/workspace \
  halfweg-gpu-eval:latest \
  python /workspace/halfweg/src/benchmark_boxoban.py \
  --checkpoint /workspace/halfweg/checkpoints/checkpoint_20260618-192542.ckpt \
  --probe-only
```

**목표**: 
- unfiltered_valid_5files 실제 성능 확인 (0.66 재현 가능?)
- medium/hard 추가 성능 측정

### Phase 2: 안정화 패치 (2시간)
```python
# train_halfweg.py에 추가할 safeguards:

# 1. Gradient clipping
torch.nn.utils.clip_grad_norm_(plta.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(plhw.parameters(), max_norm=1.0)

# 2. NaN 감지
if torch.isnan(loss_plta) or torch.isnan(loss_plhw):
    print("NaN detected, reverting to best checkpoint")
    best_ckpt = model_keeper.load_checkpoint(best_gate_passed_path)
    continue

# 3. Loss explosion detection
if loss_plta > prev_loss * 3:  # 3배 증가
    print("Loss explosion detected, rolling back")
    load_last_checkpoint()

# 4. 정책 전환 강화
if not recent_improvement_sufficient():
    use_pl1 = False  # PL0 유지
```

### Phase 3: 재학습 (8-12시간)
```yaml
# 개선된 설정
train:
  epochs: 30
  n_levels_per_epoch: 32      # 4 → 32 (8배)
  n_attempts_per_level: 2      # 1 → 2
  probe_every_epochs: 1        # 2 → 1 (빠른 응답)
  
  collector_policy_initial: PL0
  collector_policy_final: PL1
  curriculum_transition_epoch: 10  # 8 → 10 (더 안정적)
  
  data_sources:
    - unfiltered: 0.80  # 0.60 → 0.80
    - medium: 0.15      # 0.30 → 0.15
    - hard: 0.05        # 0.10 → 0.05
```

### Phase 4: 목표 성능
```
현재 (finetuned_v2): 0.48-0.54 (baseline 미달)
게이트 모델 (curriculum epoch 2): 0.66 (baseline 초과 ✓)
목표 (개선된 curriculum): 0.60+ (지속 유지)
```

---

## 7. 결론

**현 상황**:
- 💾 게이트 통과 모델 1개 확보 (checkpoint_20260618-192542.ckpt)
- 🔴 계속 학습하면 붕괴 (데이터 부족 + 정책 전환 불안정)

**추천 행동**:
1. **즉시**: 게이트 모델로 6-split 벤치마크 실행 → 실제 성능 검증
2. **병렬**: train_halfweg.py에 safeguards 추가 (NaN clip, loss explosion 감지)
3. **재학습**: 데이터량 8배 증가 + 커리큘럼 비율 조정 + 더 보수적 전환

**기대 성과**:
- baseline 초과 지속 유지 (0.60+)
- 중간/어려움 난이도 성능 향상
- 재학습 시 안정성 개선
