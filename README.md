# 실행방법

```bash
docker-compose up -d --build # 5~10분가량 소요
python sokoban_solver.py
```
모델 병렬 가동 시 ``python sokoban_solver_parallel.py``로 실행

실행 예제 로그
```bash
(sokoban_env) PS D:\projects\ML-2601-T08> python .\sokoban_solver.py
[Master] 시스템 초기화 중... 도커 컨테이너 상태를 확인합니다.
time="2026-06-20T20:55:36+09:00" level=warning msg="The \"PYTHONPATH\" variable is not set. Defaulting to a blank string."
time="2026-06-20T20:55:36+09:00" level=warning msg="D:\\projects\\ML-2601-T08\\docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion"
[+] up 2/2
 ✔ Container drc_solver_env     Running                                                                             0.0s
 ✔ Container thinker_solver_env Running                                                                             0.0s
[Master] 모든 모델 데몬 대기 완료. 시스템을 시작합니다.

[Master] 평가할 맵 경로 (종료 'q'): boxoban-levels\testsokoban.txt
[DRC33] 도커 내부로 파일 복사 완료 (testsokoban.txt)
[DRC33] 추론 연산 진행 중... (대기)
[DRC33] 완료 | 총 소요시간: 2294.28ms | 파일: testsokoban.txt
[thinker] 도커 내부로 파일 복사 완료 (testsokoban.txt)
[thinker] 추론 연산 진행 중... (대기)
[thinker] 완료 | 총 소요시간: 39656.03ms | 파일: testsokoban.txt

[Master] 평가 완료. 자동 분석 스크립트를 실행합니다...
--------------------------------------------------------------------------------
Model Name      | Acc (%)  | Solved   | Unique   | Avg Steps  | Avg Time(ms)
--------------------------------------------------------------------------------
DRC33           | 100.00   | 4        | 0        | 22.50      | 477.66
thinker         | 100.00   | 4        | 0        | 23.00      | 9045.42
--------------------------------------------------------------------------------
Total Combined Solved: [4/4]
--------------------------------------------------------------------------------

[Master] 평가할 맵 경로 (종료 'q'): boxoban-levels\testsokoban2.txt
[DRC33] 도커 내부로 파일 복사 완료 (testsokoban2.txt)
[DRC33] 추론 연산 진행 중... (대기)
[DRC33] 완료 | 총 소요시간: 5409.00ms | 파일: testsokoban2.txt
[thinker] 도커 내부로 파일 복사 완료 (testsokoban2.txt)
[thinker] 추론 연산 진행 중... (대기)
[thinker] 완료 | 총 소요시간: 233590.34ms | 파일: testsokoban2.txt

[Master] 평가 완료. 자동 분석 스크립트를 실행합니다...
--------------------------------------------------------------------------------
Model Name      | Acc (%)  | Solved   | Unique   | Avg Steps  | Avg Time(ms)
--------------------------------------------------------------------------------
DRC33           | 70.00    | 7        | 1        | 22.00      | 310.67
thinker         | 60.00    | 6        | 0        | 21.50      | 7044.27
--------------------------------------------------------------------------------
Total Combined Solved: [7/10]
--------------------------------------------------------------------------------

[Master] 평가할 맵 경로 (종료 'q'): q

[Master] 시스템을 종료합니다.
```


# AI 기반 고속 소코반 솔버 연구 (AI-based High-speed Sokoban Solver)

본 프로젝트는 PSPACE-완전 문제인 소코반(Sokoban) 퍼즐을 효율적이고 빠르게 해결하기 위한 AI 기반 솔버 연구 프로젝트입니다. 전통적인 휴리스틱 솔버(FestiVal)보다 높은 성능과 속도를 목표로 다양한 심층 강화학습 모델을 구현하고 비교 분석합니다.

## 👥 팀원 및 역할 (Team 8)
- **공통**: DataSet Encoder Design, Harness Design
- **박주안**: CNN+RNN 기반 **DRC** 파생 모델 구현
- **김재민**: CNN+RNN 기반 **Thinker** 파생 모델 구현
- **박경수**: HRL(Hierarchical Reinforcement Learning) 기반 **HalfWeg** 파생 모델 구현

## 🎯 프로젝트 목표
- **Success Criteria**: 50%의 테스트 레벨(Boxoban)에서 전통적인 Heuristic Solver(FestiVal)보다 빠른 속도로 해결
- **핵심 문제 해결**:
  - **긴 계획 지평(Long-horizon Planning)**: 수백 번의 이동이 필요한 퍼즐 해결
  - **공간 추론 및 기억**: CNN을 통한 상태 파악 및 RNN을 통한 맥락 유지
  - **계층적 전략**: 하위 목표(Sub-goals) 설정을 통한 탐색 효율 극대화

## 🧠 모델 아키텍처 (Model Architecture)

본 연구에서는 세 가지 주요 모델을 독립적으로 구현하여 성능을 측정합니다.

### 1. Thinker Model (CNN + RNN)
- **특징**: "Learning to Plan and Act" (Chung et al. 2023)
- **Key Idea**: DRC 모델에 **Imaginary Steps** (Imaginary Action + Reset Action)를 추가하여 결정론적 알고리즘의 기능을 신경망으로 구현

### 2. DRC Model (Deep Repeating ConvLSTM)
- **특징**: "Deep Repeating ConvLSTM" (Guez et al. 2019)
- **Key Idea**: 반복적인 합성곱 LSTM 구조를 통해 공간적 특징과 순차적 기억을 결합하여 복잡한 상태를 추론

### 3. HalfWeg Model (HRL)
- **특징**: "Hierarchical Reinforcement Learning" (Sergey Pastukhov, 2025)
- **Key Idea**: **6단계 정책 계층(6-Level Policy Hierarchy)**을 통한 랜드마크 기반 하위 목표 생성
  - 최상위 레벨에서 거시적인 랜드마크 설정
  - 하위 레벨에서 재귀적으로 세부 목표 및 원초적 행동(URDL) 생성

## 🛠️ 주요 기술 및 최적화 기법
- **학습 안정화**: TD Error 기반 **우선순위 기반 레벨 재생(Prioritized Level Replay)**, 이점 정규화(Advantage Normalization)
- **성능 개선**: 커리큘럼 학습(Curriculum Learning)을 통한 점진적 난이도 상향
- **연산 최적화**: 모델 경량화(Pruning/Quantization), 비동기적 추론(Asynchronous Inference)
- **모델 통합**: 공통 잠재 공간(Unified Latent Space) 임베딩 및 어텐션(Attention) 레이어 도입

## 📅 프로젝트 일정
- **10주차**: 데이터셋 인코딩 방법 조사 및 하네스(Harness) 개발
- **11주차**: AI 모델별 개별 구현
- **12주차**: 파인 튜닝(Fine-tuning)
- **13주차**: 중간 발표
- **14주차**: 결과 분석 및 성능 개선
- **15주차**: 최종 발표 준비 및 점검
- **16주차**: 최종 발표

## 📚 참고 문헌
- **Thinker**: [https://arxiv.org/abs/2307.14993](https://arxiv.org/abs/2307.14993)
- **DRC**: [https://arxiv.org/abs/1901.03559](https://arxiv.org/abs/1901.03559)
- **Boxoban Dataset**: [https://github.com/google-deepmind/boxoban-levels](https://github.com/google-deepmind/boxoban-levels)
- **FestiVal (Heuristic Solver)**: [https://ieee-cog.org/2020/papers/paper_44.pdf](https://ieee-cog.org/2020/papers/paper_44.pdf)