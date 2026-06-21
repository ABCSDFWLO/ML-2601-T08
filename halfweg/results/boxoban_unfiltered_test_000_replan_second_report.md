# Halfweg 2차 보고서: 실시간 재계획(Closed-loop Replanning) 도입 결과

## 1. 작업 범위
- 요청 범위에 따라 실시간 재계획 메커니즘만 우선 적용.
- 동일 입력 파일 `boxoban-levels/unfiltered/test/000.txt`로 재평가 수행.
- 결과를 baseline(one_shot quick)과 비교.

## 2. 코드 변경 사항

### 2.1 평가 메서드 추가
- 파일: `halfweg/src/hw_impl/evaluate.py`
- 추가된 핵심:
  - `closed_loop_replan` 메서드 분기 추가
  - 상태 불일치 기반 재계획 트리거 함수 `_compute_state_mismatch()` 추가
  - 재계획 실행 루프 `_solve__closed_loop_replan()` 추가

재계획 동작:
- 현재 상태에서 계획을 다시 생성
- 최대 `replan_every_actions`만큼 실행 후 재계획
- 상태 mismatch가 `replan_mismatch_threshold`를 넘으면 즉시 재계획

### 2.2 실행 시간 안정화 패치
- 초기 재계획 버전이 매우 무거워지는 문제를 완화하기 위해, 게임당 다중 목표 상태 전부를 순회하지 않고 대표 목표 1개만 사용하도록 조정.
- 이는 재계획 메커니즘 자체는 유지하면서 실행 가능성을 확보하기 위한 최소 변경.

### 2.3 재계획 평가 설정 파일
- 파일: `halfweg/configs/evaluate_boxoban_unfiltered_test_000_gpu_replan_quick.yaml`
- 최종 사용 파라미터:
  - `method: closed_loop_replan`
  - `n_games_to_solve: 100`
  - `replan_every_actions: 32`
  - `replan_mismatch_threshold: 0.2`

## 3. 실험 조건
- 환경: Docker + GPU (`halfweg-gpu-eval`)
- 체크포인트: `trained_models/boxoban_vast_v4_20250421-073320__repacked.ckpt`
- 레벨 입력: `/workspace/boxoban-levels/unfiltered/test/000.txt`
- 정책: `last` (`Policy5 (layer_idx=6)`)

로그 파일:
- 재계획 런: `halfweg/results/boxoban_unfiltered_test_000_eval_replan_quick.log`
- baseline 런: `halfweg/results/boxoban_unfiltered_test_000_eval_quick.log`

## 4. 결과 비교 (100 games)

### 4.1 Baseline (one_shot)
- `solved_mean`: `0.61`
- `mse_mean`: `4.704882831167356`
- `proposed_plan_length_mean`: `98.82785808147175`
- `solved_plan_length_mean`: `102.37704918032787`
- `Solving`: `18.73796 sec`

### 4.2 Replanning (closed_loop_replan)
- `solved_mean`: `0.19`
- `mse_mean`: `5.237627522754686`
- `proposed_plan_length_mean`: `123.39`
- `solved_plan_length_mean`: `39.578947368421055`
- `Solving`: `58.53352 sec`

## 5. 해석
- 이번 구현/파라미터 조합에서 실시간 재계획은 성능 개선에 실패했고, 오히려 성능이 크게 하락.
- `solved_mean`은 `0.61 -> 0.19`로 감소.
- `mse_mean`은 증가(`4.70 -> 5.24`)하여 최종 상태 정합도도 악화.
- 평균 제안 계획 길이와 런타임이 크게 증가하여 계산 비용도 상승.

가능한 원인:
- 재계획 트리거가 환경 동역학 및 정책 품질과 정합되지 않아 계획이 흔들림.
- 대표 목표 1개 사용 단순화가 목표 다양성을 줄여 성능 저하를 유발 가능.
- 현재 mismatch 정의(단순 평균 절대오차)가 소코반 실행 가능성과 직접적으로 일치하지 않을 수 있음.

## 6. 결론
- 실시간 재계획을 "도입"하고 동일 조건 테스트를 완료했으며, 현재 구현 상태에서는 `solved_mean` 개선이 확인되지 않음.
- 현 버전은 2차 실험 결과 기준으로 baseline 대비 회귀(regression)로 판단.

## 7. 후속 실험 제안 (재계획 단독 범위 내)
- mismatch 지표를 박스 위치 중심 오차로 재정의(플레이어 위치 영향 분리).
- `replan_every_actions`와 threshold의 조합 스윕(예: 16/24/32, 0.15/0.2/0.25).
- 재계획 조건에 "박스 진전 없음" 기준 추가(박스 상태가 일정 step 동안 동일할 때만 강제 재계획).