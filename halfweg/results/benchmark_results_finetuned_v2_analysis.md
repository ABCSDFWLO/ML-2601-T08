# Benchmark Analysis: finetuned_v2

## Summary
이번 `finetuned_v2` 체크포인트는 이전 실패한 finetune 대비 성능을 크게 회복했다. 다만 원본 baseline 체크포인트와 비교하면, 주력 분포인 `unfiltered_*`에서는 아직 소폭 뒤처지고 `hard_all`은 완전히 회복하지 못했다.

핵심 해석은 다음과 같다.

- 학습 파이프라인 복구는 성공했다.
- 이전 finetune collapse는 해소됐다.
- 하지만 현재 finetune은 baseline을 대체할 수준은 아니다.
- probe 0.40은 유효한 개선 신호였지만, 전체 split 일반화로는 과대평가였다.

## Compared Files
- Baseline: `halfweg/results/benchmark_results.csv`
- Previous failed finetune: `halfweg/results/benchmark_results_finetuned.csv`
- Current fixed finetune: `halfweg/results/benchmark_results_finetuned_v2.csv`

## Solve Rate Comparison

| split | baseline | failed finetune | finetuned_v2 | vs baseline | vs failed finetune |
|---|---:|---:|---:|---:|---:|
| unfiltered_test | 0.60 | 0.08 | 0.48 | -0.12 | +0.40 |
| unfiltered_valid_5files | 0.58 | 0.02 | 0.54 | -0.04 | +0.52 |
| unfiltered_train_10files | 0.52 | 0.02 | 0.50 | -0.02 | +0.48 |
| medium_valid_5files | 0.08 | 0.00 | 0.09 | +0.01 | +0.09 |
| medium_train_5files | 0.10 | 0.00 | 0.06 | -0.04 | +0.06 |
| hard_all | 0.05 | 0.00 | 0.00 | -0.05 | +0.00 |

## What Improved

### 1. Collapse recovery is real
이전 finetune은 `unfiltered_valid_5files=0.02`, `unfiltered_train_10files=0.02`, `unfiltered_test=0.08` 수준으로 사실상 붕괴 상태였다. 이번 run은 이를 각각 `0.54`, `0.50`, `0.48`까지 회복했다.

이건 단순 노이즈가 아니라 학습 신호 복구의 결과로 보는 것이 맞다.

### 2. Medium valid는 baseline을 소폭 초과
`medium_valid_5files`가 `0.08 -> 0.09`로 아주 작지만 baseline을 넘었다. 규모는 작지만, partial trajectory 활용과 collector policy 안정화가 어려운 분포에 일부 도움을 준 신호로 볼 수 있다.

### 3. Timeout rate는 medium/hard에서 개선
baseline 대비 timeout rate 변화:

- `medium_valid_5files`: `0.7639 -> 0.6875`
- `medium_train_5files`: `0.7642 -> 0.6871`
- `hard_all`: `0.7156 -> 0.6736`

즉, 더 자주 끝까지 진행은 하지만, 그 진행이 곧 해결로 이어지지는 않았다. `hard_all`에서 solve rate가 0인 점이 그 증거다.

## What Regressed

### 1. Main distribution still below baseline
가장 중요한 건 `unfiltered_*` 3개 split이다. 이 영역은 실제 주된 데이터 분포에 가깝고 샘플 수도 많다. 여기서 모두 baseline보다 낮다.

- `unfiltered_test`: `0.60 -> 0.48`
- `unfiltered_valid_5files`: `0.58 -> 0.54`
- `unfiltered_train_10files`: `0.52 -> 0.50`

차이는 크지 않지만, baseline이 여전히 최고 성능 체크포인트라는 뜻이다.

### 2. Hard split generalization failed
`hard_all`은 baseline도 낮았지만 최소한 `0.05`는 해결했다. 반면 `finetuned_v2`는 `0.00`이다. 현재 finetune이 harder search regime을 약화시켰다고 보는 편이 맞다.

### 3. Probe score overestimated global gain
학습 중 최고 probe는 `0.40`이었다. 그런데 full benchmark에서는 다음과 같이 나왔다.

- `unfiltered_test`: `0.48`
- `unfiltered_valid_5files`: `0.54`
- `unfiltered_train_10files`: `0.50`
- `medium/hard`: 여전히 약함

즉 probe는 “학습이 살아났는지” 확인하는 용도로는 유효했지만, “전체 일반화가 baseline을 넘었는지” 판단하는 지표로는 부족했다.

## Secondary Metric Reading

### Replan count
`unfiltered_*`에서 평균 replan count는 baseline 대비 소폭 증가했다.

- unfiltered_test: `8.75 -> 8.90`
- unfiltered_valid_5files: `9.93 -> 10.09`
- unfiltered_train_10files: `10.59 -> 11.59`

이는 정책이 baseline보다 덜 안정적이어서 더 자주 재계획이 필요하다는 신호다.

### Box push count
`unfiltered_*`에서는 평균 push 수가 baseline보다 감소했다.

- unfiltered_test: `15.24 -> 14.39`
- unfiltered_valid_5files: `17.15 -> 14.09`
- unfiltered_train_10files: `18.35 -> 13.96`

이 수치는 무조건 좋은 값은 아니다. 더 적은 push로 푼 것이 아니라, 더 이른 중단 또는 더 약한 탐색으로 인해 puzzle interaction 자체가 줄었을 가능성이 있다.

### MSE
MSE는 이전 failed finetune 대비 크게 개선되었고 baseline에도 근접했다.

- unfiltered_test: `8.46 -> 5.55` (failed 대비 큰 개선)
- unfiltered_valid_5files: `8.79 -> 5.41`
- unfiltered_train_10files: `8.84 -> 5.74`

다만 baseline 대비 완전 우세는 아니다. 표현 품질 회복은 됐지만 행동 정책 우위까지는 이어지지 않았다.

## Overall Verdict

### Good news
- training pipeline repair는 성공
- previous failed finetune 대비 대폭 회복
- PLHW/PLTA 모두 실제로 다시 학습함
- medium_valid에서 소폭 개선 신호 확인

### Bad news
- baseline을 아직 못 넘음
- hard split은 악화
- probe 0.40만 보고 best model로 채택하면 과신 위험이 있음

## Root-Cause Interpretation
현재 결과는 다음 가설과 일치한다.

1. `collector_policy=PL0`는 학습 신호를 안정화하는 데는 효과적이었다.
2. 하지만 너무 보수적인 trajectory 분포를 만들어 baseline이 가진 더 강한 planning behavior 일부를 약화시켰다.
3. 따라서 “붕괴 복구”에는 성공했지만 “기존 최고 모델 초과”에는 실패했다.
4. probe set이 단일 파일 기반이라 best-checkpoint 선택이 과적합 방향으로 치우쳤다.

## Recommended Next Steps

### 1. Best checkpoint selection 기준을 바꿔야 함
단일 `probe_levels=/workspace/boxoban-levels/unfiltered/valid/000.txt` 대신 3~10개 파일 묶음 평균으로 선택해야 한다.

우선순위가 가장 높다.

### 2. Collector policy를 PL0 고정으로 두지 말 것
다음 2개 실험이 바로 필요하다.

- `collector_policy=PL1`
- epoch 초반 PL0, 후반 PL1 curriculum

현재 결과만 보면 PL0는 회복용으로는 좋지만 최종 generalization에는 너무 약하다.

### 3. Hard/medium 비중을 명시적으로 높일 것
현재 train source가 `unfiltered/train` 중심이라 harder distribution transfer가 약하다. mixed curriculum 또는 split-weighted sampling이 필요하다.

### 4. Best-vs-baseline gate를 추가할 것
새 finetune checkpoint를 채택하기 전에 최소한 `unfiltered_valid_5files`와 `unfiltered_train_10files`에서 baseline 이상일 때만 승격하는 gate가 필요하다.

## Final Conclusion
이번 결과는 “실패한 finetune 복구”로 보면 성공이다. 하지만 “baseline을 넘는 새 주력 모델 확보”로 보면 아직 실패다.

따라서 현재 최선의 운영 판단은 다음과 같다.

- 배포/대표 체크포인트는 여전히 baseline 유지
- 이번 `finetuned_v2`는 recovery proof + next iteration 출발점으로 사용
- 다음 실험의 핵심은 `multi-file probe`, `PL1 or curriculum collector`, `harder split mixing`
