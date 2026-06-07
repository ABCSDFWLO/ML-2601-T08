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

## HalfWeg 구현 실행 가이드

본 저장소에는 arXiv:2504.04366v1 논문 아이디어를 반영한 HalfWeg 스타일 구현이 포함되어 있습니다.

- 핵심 모듈: `halfweg/`
- 학습 스크립트: `scripts/train_halfweg.py`
- 단일 레벨 추론 스크립트: `scripts/solve_level.py`

### 1) 의존성 설치

```bash
pip install -r requirements.txt
```

### 2) 학습

```bash
python scripts/train_halfweg.py \
  --dataset-root boxoban-levels \
  --split-glob "unfiltered/train/*.txt" \
  --epochs 3 \
  --device cpu \
  --save checkpoints/halfweg.pt
```

### 3) 단일 레벨 추론

```bash
python scripts/solve_level.py \
  --checkpoint checkpoints/halfweg.pt \
  --level-file boxoban-levels/unfiltered/test/000.txt \
  --level-index 0 \
  --player-mode all \
  --searches 10 \
  --device cpu
```

### 구현된 핵심 요소

- Sokoban planning tuple (u, v, b)
- 두 모델 구조: Action model MA, State model MS
- 재귀 정책 PL0..PLR
- PL0 exhaustive search + 상위 레벨 two-leg search
- self-play 기반 리플레이 버퍼 샘플링
- 계획 데이터셋 생성 후 MA/MS 공동 학습

참고: 본 구현은 연구 재현용 baseline 코드이며, 논문 수치(해결률) 재현을 위해서는 더 큰 학습량, 하이퍼파라미터 튜닝, 분산 학습/검색 자원 확장이 필요합니다.

## Docker 실행 가이드 (권장)

환경 호환성을 위해 Docker 기반 실행을 지원합니다.

### 1) 이미지 빌드

```bash
docker compose build
```

### 2) 학습 실행

```bash
docker compose run --rm halfweg \
  python scripts/train_halfweg.py \
  --dataset-root boxoban-levels \
  --split-glob "unfiltered/train/*.txt" \
  --epochs 3 \
  --device cpu \
  --save checkpoints/halfweg.pt
```

### 3) 단일 레벨 추론

```bash
docker compose run --rm halfweg \
  python scripts/solve_level.py \
  --checkpoint checkpoints/halfweg.pt \
  --level-file boxoban-levels/unfiltered/test/000.txt \
  --level-index 0 \
  --player-mode all \
  --searches 10 \
  --device cpu
```

### 참고

- 컨테이너 내부 작업 디렉터리는 `/workspace`입니다.
- 루트 폴더가 볼륨 마운트되므로 학습 산출물(`checkpoints/`)은 호스트에도 그대로 저장됩니다.
- 기본 서비스(`halfweg`)는 `requirements-docker.txt`를 사용하며 CPU 전용 PyTorch wheel을 설치합니다.
- GPU 서비스(`halfweg-gpu`)는 `requirements-docker-gpu.txt`를 사용합니다.

### GPU 실행 (선택)

사전 조건:
- NVIDIA GPU + 최신 드라이버
- Docker Desktop의 GPU 지원(Windows는 WSL2 백엔드 권장)

GPU 이미지 빌드:

```bash
docker compose --profile gpu build halfweg-gpu
```

GPU 학습 실행:

```bash
docker compose --profile gpu run --rm halfweg-gpu \
  python scripts/train_halfweg.py \
  --dataset-root boxoban-levels \
  --split-glob "unfiltered/train/*.txt" \
  --epochs 3 \
  --device cuda \
  --save checkpoints/halfweg_gpu.pt
```

GPU 추론 실행:

```bash
docker compose --profile gpu run --rm halfweg-gpu \
  python scripts/solve_level.py \
  --checkpoint checkpoints/halfweg_gpu.pt \
  --level-file boxoban-levels/unfiltered/test/000.txt \
  --level-index 0 \
  --player-mode all \
  --searches 10 \
  --device cuda
```