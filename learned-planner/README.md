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
