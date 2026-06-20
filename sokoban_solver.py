import sys
import subprocess
import time

# 각 모델별 하위 폴더 경로 매핑
MODEL_ROUTING = {
    "drc": "learned-planner/drc_solver.py",
    # 향후 추가될 모델 경로
    # "thinker": "thinker/thinker_solver.py" 
}

def run_model_solver_sync(model_name, file_path):
    if model_name not in MODEL_ROUTING:
        print(f"[{model_name.upper()}] 오류: 라우팅 경로가 정의되지 않았습니다.")
        return
        
    solver_script = MODEL_ROUTING[model_name]
    
    try:
        # 블로킹 실행 및 실시간 터미널 출력 허용 (capture_output 제거)
        subprocess.run(
            ["python", solver_script, file_path],
            check=False,
            timeout=300
        )
            
    except subprocess.TimeoutExpired:
        print(f"[{model_name.upper()}] 타임아웃 발생 (300초 초과): {file_path}")
        subprocess.run(["docker", "restart", f"{model_name}_solver_env"])

def main():
    models = ["drc"] # 활성화된 모델 목록
    
    print("[Master] 시스템 초기화 중... 도커 컨테이너 상태를 확인합니다.")
    subprocess.run(["docker-compose", "up", "-d"])
    time.sleep(2)
    print("[Master] 모든 모델 데몬 대기 완료.")

    try:
        while True:
            file_path = input("\n[Master] 평가할 맵 경로 (종료 'q'): ").strip()
            
            if file_path.lower() in ['q', 'quit', 'exit']:
                break
                
            if not file_path:
                continue

            for model in models:
                run_model_solver_sync(model, file_path)
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[Master] 시스템을 종료합니다.")

if __name__ == "__main__":
    main()