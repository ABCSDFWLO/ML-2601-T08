import sys
import subprocess
import time
import os

MODEL_ROUTING = {
    "drc": "learned-planner/drc_solver.py",
    "thinker": "thinker/thinker_solver.py",
    "halfweg": "halfweg/halfweg_solver.py"
}

def run_model_solver_sync(model_name, file_path):
    if model_name not in MODEL_ROUTING:
        print(f"[{model_name.upper()}] 오류: 라우팅 경로가 정의되지 않았습니다.")
        return None
        
    solver_script = MODEL_ROUTING[model_name]
    json_path = None
    
    try:
        process = subprocess.Popen(
            ["python", "-u", solver_script, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("[JSON_OUTPUT]"):
                json_path = line.replace("[JSON_OUTPUT]", "").strip()
            else:
                print(line)
                
        process.wait(timeout=300)
        return json_path
            
    except subprocess.TimeoutExpired:
        print(f"[{model_name.upper()}] 타임아웃 발생 (300초 초과): {file_path}")
        subprocess.run(["docker", "restart", f"{model_name}_solver_env"])
        return None

def main():
    models = ["drc", "thinker", "halfweg"] 
    
    print("[Master] 시스템 초기화 중... 도커 컨테이너 상태를 확인합니다.")
    subprocess.run(["docker-compose", "up", "-d"])
    time.sleep(2)
    print("[Master] 모든 모델 데몬 대기 완료. 시스템을 시작합니다.")

    # 저장소 폴더 강제 생성
    os.makedirs("solutions", exist_ok=True)

    try:
        while True:
            file_path = input("\n[Master] 평가할 맵 경로 (종료 'q'): ").strip()
            
            if file_path.lower() in ['q', 'quit', 'exit']:
                break
                
            if not file_path:
                continue

            generated_jsons = []
            
            for model in models:
                json_result = run_model_solver_sync(model, file_path)
                if json_result and os.path.exists(json_result):
                    generated_jsons.append(json_result)
            
            if generated_jsons:
                print(f"\n[Master] 평가 완료. 자동 분석 스크립트를 실행합니다...")
                analyze_cmd = ["python", "analyze_model_results.py"] + generated_jsons
                subprocess.run(analyze_cmd)
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[Master] 시스템을 종료합니다.")

if __name__ == "__main__":
    main()