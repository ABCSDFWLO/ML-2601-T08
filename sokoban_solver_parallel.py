import sys
import subprocess
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor

# 원자적 출력을 위한 락
print_lock = threading.Lock()

def safe_print(message):
    with print_lock:
        print(message, flush=True)

# 모델별 실행 스크립트 경로 매핑
MODEL_ROUTING = {
    "drc": "learned-planner/drc_solver.py",
    "thinker": "thinker/thinker_solver.py"
}

def run_model_solver_async(model_name, file_path):
    """비동기적으로 개별 모델 솔버를 실행하고 결과 경로를 반환합니다."""
    if model_name not in MODEL_ROUTING:
        safe_print(f"[{model_name.upper()}] 오류: 라우팅 경로가 없습니다.")
        return None
        
    solver_script = MODEL_ROUTING[model_name]
    json_path = None
    
    try:
        # 하위 프로세스 실행 (-u: 버퍼링 해제)
        process = subprocess.Popen(
            ["python", "-u", solver_script, file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        # 실시간 출력 캡처
        for line in process.stdout:
            line = line.strip()
            if not line: continue
            
            # JSON 반출 경로 포착 (화면 출력 생략)
            if line.startswith("[JSON_OUTPUT]"):
                json_path = line.replace("[JSON_OUTPUT]", "").strip()
            else:
                safe_print(line)
                
        process.wait(timeout=600) # Thinker 등을 고려해 타임아웃 넉넉히 설정
        return json_path
            
    except subprocess.TimeoutExpired:
        safe_print(f"[{model_name.upper()}] 타임아웃 발생: {file_path}")
        subprocess.run(["docker", "restart", f"{model_name}_solver_env"])
        return None

def main():
    # 병렬로 실행할 모델 목록
    models = ["drc", "thinker"]
    
    safe_print("[Master] 시스템 초기화 중... 도커 컨테이너 상태를 확인합니다.")
    subprocess.run(["docker-compose", "up", "-d"])
    time.sleep(2)
    safe_print("[Master] 모든 모델 데몬 대기 완료.")

    # 결과물 저장을 위한 디렉터리 생성
    os.makedirs("solutions", exist_ok=True)

    try:
        while True:
            file_path = input("\n[Master] 평가할 맵 경로 (종료 'q'): ").strip()
            
            if file_path.lower() in ['q', 'quit', 'exit']:
                break
                
            if not file_path or not os.path.exists(file_path):
                safe_print("[Master] 유효하지 않은 경로입니다.")
                continue

            generated_jsons = []
            
            # ThreadPoolExecutor를 이용해 모델별 작업을 병렬로 제출
            with ThreadPoolExecutor(max_workers=len(models)) as executor:
                future_to_model = {executor.submit(run_model_solver_async, m, file_path): m for m in models}
                
                for future in future_to_model:
                    json_result = future.result()
                    if json_result and os.path.exists(json_result):
                        generated_jsons.append(json_result)
            
            # 모델들의 연산이 모두 끝난 후 분석 실행
            if generated_jsons:
                safe_print("\n[Master] 모든 모델의 연산 완료. 자동 분석 스크립트를 실행합니다...")
                subprocess.run(["python", "analyze_model_results.py"] + generated_jsons)
                
    except KeyboardInterrupt:
        pass
    finally:
        safe_print("\n[Master] 시스템을 종료합니다.")

if __name__ == "__main__":
    main()