# Docker 실행 방법

## 1. Docker 초기화 예제
D:\projects\ML-2601-T08\thinker를 자신의 프로젝트 경로에 맞게 수정할 것.
```
cd D:\projects\ML-2601-T08\thinker

docker build -t thinker_env:v1 .

docker run --gpus all --shm-size=8g -p 8265:8265 -it -v "D:\projects\ML-2601-T08\thinker:/workspace" thinker_env:v1 bash
```
**주의:** --shm-size=8g는 컴퓨터 RAM 용량에 따라 조절할 것. 1/3 크기 권장.

## 2. 도커 실행 이후 환경 초기화

```
cd /workspace/csokoban
pip install -e .

cd /workspace/thinker
python setup.py build_ext --inplace

export PYTHONPATH="/workspace/csokoban:$PYTHONPATH"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export GRPC_POLL_STRATEGY=poll
export GRPC_ENABLE_FORK_SUPPORT=1
```

## 3. json solution 도출 예제
```
python inference.py --load_checkpoint /workspace/trained/trained_backup/base --map_path /workspace/testsokoban.txt --greedy --rec_t 20
```
``--map_path``: docker workspace 내에 존재하는 boxoban 포맷 파일
``--rec_t``: 시뮬레이션 깊이. 클수록 정확해지겠으나, 더 느려짐.

test 및 train 예제
```
python test.py --load_checkpoint /workspace/trained/trained_backup/base --env cSokoban-v0 --test_eps_n 100 --ray_gpu 1 --ray_mem 0 \ test_log.txt 2>&1 & tail -f test_log.txt

python train.py --load_checkpoint /workspace/trained/trained_backup/base --env cSokoban-v0 --test_eps_n 100 --ray_gpu 1 --ray_mem 0
```


# Thinker: Learning to Plan and Act

This is the official repository for the paper titled "Thinker: Learning to Plan and Act". 

## Prerequisites

Ensure that Pytorch is installed (versions v2.0.0 and v1.13.0 have been tested).

## Installation

1. Update and install the necessary packages:

```bash
sudo apt-get update
sudo apt-get install python-opencv build-essential -y
pip install -r requirement.txt
```

2. Install the C++ version of Sokoban (skip this step if you're not running experiments on Sokoban):
```bash
cd csokoban
pip install -e .
```

3. Compile the Cython version of the wrapped environment:
```bash
cd thinker
python setup.py build_ext --inplace
```

## Usage
Run the following commands in the `thinker` directory:

Atari default run (change the environment if needed):

```bash
python train.py --env BreakoutNoFrameskip-v4
```

Raw MDP Atari default run:

```bash
python train.py --env BreakoutNoFrameskip-v4 --disable_model --unroll_length 20 --learning_rate 0.0003
```

Sokoban default run:

```bash
python train.py --env cSokoban-v0 --model_size_nn 1 --disable_frame_copy --discounting 0.97 --reward_clip -1
```

Raw MDP Sokoban default run:

```bash
python train.py --env cSokoban-v0 --model_size_nn 1 --disable_frame_copy --discounting 0.97 --reward_clip -1 --disable_model --unroll_length 20 --learning_rate 0.0003
```


The number of GPUs will be detected automatically. A single RTX3090 is sufficient for the Sokoban default run, while two RTX3090s are required for the Atari default run. The number of CPUs and GPUs allocated can be controlled with the `--ray_cpu` and `--ray_gpu` options, e.g., `--ray_cpu 16 --ray_gpu 1` limits usage to 16 CPUs and 1 GPU.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
[Redacted]
