# Halfweg 최종 실험 보고서: Closed-loop Replanning on boxoban-levels/unfiltered/test/000.txt

> 실험 날짜: 2026-06-15  
> 체크포인트: `trained_models/boxoban_vast_v4_20250421-073320__repacked.ckpt`  
> 입력: `boxoban-levels/unfiltered/test/000.txt`  
> Docker 이미지: `halfweg-gpu-eval` (pytorch 2.5.1 + CUDA 12.4)

---

## 1. 실험 요약 (3회 주요 런 비교)

| 구분 | 방법 | 게임 수 | solved_mean | mse_mean | solving time |
|---|---|---|---|---|---|
| **Baseline** | one_shot | 100 | **0.61** | 4.7049 | 18.7 sec |
| **2차 실험 (회귀)** | closed_loop_replan (단일 목표 고정 + 주기 재계획) | 100 | 0.19 | 5.2376 | 58.5 sec |
| **최종 수정 10게임** | closed_loop_replan (상자 중심 + 다중 목표 + stall 트리거) | 10 | 0.60 | 4.6890 | 458.4 sec |
| **최종 수정 100게임** | closed_loop_replan (상자 중심 + 다중 목표 + stall 트리거) | 100 | **0.65** | 5.2350 | 5139.5 sec |

---

## 2. Baseline vs 최종 수정 (100게임 기준)

| 지표 | Baseline (one_shot) | Replanning (최종) | 변화 |
|---|---|---|---|
| `solved_mean` | 0.61 | **0.65** | **+0.04 ▲** |
| `mse_mean` | 4.7049 | 5.2350 | +0.53 ▲ |
| `proposed_plan_length_mean` | 98.83 | 71.25 | -27.58 ▼ |
| `solved_plan_length_mean` | 102.38 | **47.52** | **-54.86 ▼** |
| `games_cnt` | 100 | 100 | - |
| solving time | 18.7 sec | 5139.5 sec | ×275 |

---

## 3. 최종 적용 코드 변경 내용

### 3.1 파일: `halfweg/src/hw_impl/evaluate.py`

**A. 상자 중심 mismatch 지표 전환**
```python
def _compute_state_mismatch(curr_state, target_state):
    # 플레이어 채널 무시, 상자 채널(2)만 비교
    if curr_state.shape[0] > 2 and target_state.shape[0] > 2:
        return float(np.mean(np.abs(curr_state[2] - target_state[2])))
    return float(np.mean(np.abs(curr_state - target_state)))
```

**B. 다중 목표 순회 복원**
- 이전 단순화 패치(대표 1개 고정)를 제거하고 `for target_i in range(len(curr_targets_t)):`로 복원

**C. 박스 stall 기반 재계획 트리거**
- 최근 `replan_stall_steps` 스텝 동안 상자가 이동하지 않으면 재계획
- `replan_every_actions: 0`으로 무조건 주기 재계획 비활성화

**D. 최소 실행 커밋 윈도우**
- `min_commit_steps`: 트리거 평가를 최소 K 스텝 이후로 지연, 과잉 재계획 억제

**E. 진행률 로그**
- 게임 단위: `[Replan] Planning Game X/N`
- 타깃 단위: `[Replan] Game X: target Y/Z`
- 결과 집계 단위: `[Progress] Processing Game X/N`

### 3.2 최종 실험 설정 파일

| 파라미터 | 최종값 |
|---|---|
| `method` | `closed_loop_replan` |
| `n_max_episode_steps` | 100 |
| `replan_every_actions` | 0 (비활성) |
| `replan_mismatch_threshold` | 0.20 |
| `replan_stall_steps` | 10 |
| `min_commit_steps` | 4 |
| `targets` | all |

---

## 4. 분석 및 해석

### solved_mean: 0.61 → 0.65 (+0.04 개선)
- 재계획 메커니즘이 올바르게 작동하면, 기존 one_shot보다 성공률이 소폭 상승 가능함을 검증.
- 박스 상태 고착 시 재계획하는 구조가 일부 퍼즐을 구제하는 데 효과가 있음.

### solved_plan_length_mean: 102.38 → 47.52 (절반 이하로 감소)
- 성공 시 훨씬 짧은 경로로 풀어냄. 재계획이 비효율적 계획 경로를 일찍 끊고 새로운 경로를 택하는 효과.

### mse_mean: 4.70 → 5.24 (소폭 증가)
- 실패 게임에서 목표 상태와의 오차가 다소 늘어남. 재계획 과정에서 플레이어가 불필요한 방향으로 유도되는 경우 포함.

### 연산 비용: ×275 증가
- 100게임에 5139초(약 85분). 다중 목표×재계획 호출 횟수가 주 원인.
- 실용적 적용을 위해 추가 최적화(배치 플래닝, 타깃 후보 사전 필터링) 필요.

---

## 5. 생성된 결과물 파일 목록

| 파일 | 설명 |
|---|---|
| `halfweg/src/hw_impl/evaluate.py` | 재계획 로직 전체 수정본 |
| `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_100games.yaml` | 최종 100게임 설정 |
| `halfweg/results/boxoban_unfiltered_test_000_eval_replan_100games.log` | 최종 100게임 완주 로그 |
| `halfweg/results/boxoban_unfiltered_test_000_eval_replan_10games.log` | 중간 10게임 완주 로그 |
| `halfweg/results/boxoban_unfiltered_test_000_eval_replan_onegame.log` | 1게임 빠른 검증 로그 |
| `halfweg/results/boxoban_unfiltered_test_000_eval_quick.log` | baseline one_shot 로그 |

---

## 6. 결론

> **실시간 재계획(Closed-loop Replanning) 메커니즘을 올바르게 적용한 결과, baseline(0.61) 대비 solved_mean이 0.65로 개선되었습니다.**
>
> 단, 연산 비용이 275배 증가하는 트레이드오프가 존재하며, 실용화를 위해서는 타깃 후보 필터링 또는 배치 추론 최적화가 필요합니다.
