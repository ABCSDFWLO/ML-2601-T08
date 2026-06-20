import sys
import subprocess
import time
import os

MODEL_ROUTING = {
    "drc": "learned-planner/drc_solver.py",
    # 향후 모델 추가 시 여기에 경로 매핑
    # "thinker": "thinker/thinker_solver.py" 
}

def run_model_solver_sync(model_name, file_path):
    if model_name not in MODEL_ROUTING:
        print(f"[{model_name.upper()}] 오류: 라우팅 경로가 정의되지 않았습니다.")
        return None
        
    solver_script = MODEL_ROUTING[model_name]
    json_path = None
    
    try:
        # 서브프로세스의 출력을 실시간으로 가로채어 메인 터미널에 스트리밍합니다.
        process = subprocess.Popen(
            ["python", solver_script, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
                
            # JSON 반출 경로 태그를 포착하면 변수에 저장하고 화면에는 출력하지 않습니다.
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
    models = ["drc"] 
    
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

            generated_jsons = []
            
            # 각 모델별 순차 평가 및 결과 수집
            for model in models:
                json_result = run_model_solver_sync(model, file_path)
                if json_result and os.path.exists(json_result):
                    generated_jsons.append(json_result)
            
            # 모델 평가가 1개 이상 완료되었을 경우 자동 분석 실행
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