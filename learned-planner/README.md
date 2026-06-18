# 실행 방법
cd learned-planner

1. docker build -t learned_planner_img .
2. docker run -it -d --gpus all --name drc_solver_env -v ${PWD}:/workspace learned_planner_img /bin/bash
 - GPU 활성화 테스트: docker exec drc_solver_env nvidia-smi 
3. 환경 세팅
 - Windows: .\init.ps1
 - MAC/LINUX: bash init.sh
4. python drc_solver.py
 - 테스트: .\testsokoban.txt

# 실행 예제

```bash
(ml_env) PS D:\projects\ML-2601-T08\learned-planner> python .\drc_solver.py
[Master] Starting Docker container...
[Master] Applying cache path patch as root...
[Master] Cleaning up previous tasks...
[Master] Starting DRC33 Daemon in background...
[Master] Waiting for DRC33 model to load into VRAM (10s)...

[Master] 평가할 윈도우 파일/폴더 경로를 입력하세요 (종료: 'q'): D:\projects\ML-2601-T08\learned-planner\testsokoban.txt
[Master] Copying data to Docker (testsokoban.txt)...
[Master] Triggering Docker daemon...
[18:42:56] [Master] Waiting for Docker to finish...
[18:43:01] [Master] Retrieving results from Docker...
[18:43:01] [Master] Done! Saved locally to: D:\projects\ML-2601-T08\learned-planner\DRC33_testsokoban.txt_results.json

[Master] 평가할 윈도우 파일/폴더 경로를 입력하세요 (종료: 'q'): q
[Master] Shutting down...

(ml_env) PS D:\projects\ML-2601-T08\learned-planner> Get-Content .\DRC33_testsokoban.txt_results.json -TotalCount 20
{
    "data": {
        "testsokoban.txt_map_000": {
            "status": "success",
            "steps": 4,
            "inference_time_ms": 1018.4422929996799,
            "total_system_time_ms": 1140.6620159996237,
            "solution": [
                {
                    "step": 1,
                    "forward_time": 0.9721962300000087,
                    "action": "R",
                    "policy_logits": [
                        -2.047409772872925,
                        -0.7584506273269653,
                        -1.4383580684661865,
                        3.6624104976654053
                    ],
                    "value": 8.233467102050781
                },
(ml_env) PS D:\projects\ML-2601-T08\learned-planner>
```

**결과 분석:** .\analyze.py <json 파일 경로>

----


## Citation

If you use this code, please cite our work:

```bibtex
@misc{taufeeque2025planningrecurrentneuralnetwork,
      title={Planning in a recurrent neural network that plays Sokoban}, 
      author={Mohammad Taufeeque and Philip Quirke and Maximilian Li and Chris Cundy and Aaron David Tucker and Adam Gleave and Adrià Garriga-Alonso},
      year={2025},
      eprint={2407.15421},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.15421}, 
}

@misc{taufeeque2025interpretinglearnedsearchfinding,
      title={Interpreting learned search: finding a transition model and value function in an RNN that plays Sokoban}, 
      author={Mohammad Taufeeque and Aaron David Tucker and Adam Gleave and Adrià Garriga-Alonso},
      year={2025},
      eprint={2506.10138},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.10138}, 
}
```
