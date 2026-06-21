import sys
import subprocess
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor

print_lock = threading.Lock()

def safe_print(message):
    with print_lock:
        print(message, flush=True)

# 모델별 실행 스크립트 경로 매핑
MODEL_ROUTING = {
    "drc": "learned-planner/drc_solver.py",
    "thinker": "thinker/thinker_solver.py",
    "halfweg": "halfweg/halfweg_solver.py"
}

def run_model_solver_async(model_name, file_path):
    if model_name not in MODEL_ROUTING:
        safe_print(f"[{model_name.upper()}] 오류: 라우팅 경로가 없습니다.")
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
            if not line: continue
            
            if line.startswith("[JSON_OUTPUT]"):
                json_path = line.replace("[JSON_OUTPUT]", "").strip()
            else:
                safe_print(line)
                
        process.wait(timeout=600)
        return json_path
            
    except subprocess.TimeoutExpired:
        safe_print(f"[{model_name.upper()}] 타임아웃 발생: {file_path}")
        if model_name in ["drc", "thinker"]:
            subprocess.run(["docker", "restart", f"{model_name}_solver_env"])
        return None

def main():
    # 시스템에 등록된 전체 모델 목록
    ALL_MODELS = list(MODEL_ROUTING.keys())
    
    safe_print("[Master] 시스템 초기화 중... 환경 상태를 확인합니다.")
    subprocess.run(["docker-compose", "up", "-d"])
    time.sleep(2)
    safe_print("[Master] 모든 모델 데몬 대기 완료.")

    os.makedirs("solutions", exist_ok=True)

    try:
        while True:
            # 입력 안내문 변경
            user_input = input("\n[Master] 실행할 모델명(선택) 및 맵 경로 (종료 'q')\n-> ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
                
            if not user_input:
                continue

            # 1. 입력 문자열 동적 파싱 로직
            tokens = user_input.split()
            target_models = []
            path_tokens = []
            
            for token in tokens:
                # 파일 경로 토큰이 아직 나오지 않았고, 토큰이 유효한 모델명일 경우
                if token.lower() in ALL_MODELS and not path_tokens:
                    target_models.append(token.lower())
                else:
                    # 유효한 모델명이 아닌 토큰이 등장한 시점부터는 모두 파일 경로로 취급
                    path_tokens.append(token)
            
            file_path = " ".join(path_tokens)
            
            # 모델을 명시하지 않은 경우 기본값으로 모든 모델 실행
            if not target_models:
                target_models = ALL_MODELS

            # 2. 경로 검증
            if not file_path or not os.path.exists(file_path):
                safe_print(f"[Master] 유효하지 않은 파일 경로입니다: {file_path}")
                continue

            generated_jsons = []
            safe_print(f"[Master] 실행 대상 모델: {', '.join(target_models).upper()}")
            
            # 3. 파싱된 타겟 모델 목록을 바탕으로 병렬 평가 제출
            with ThreadPoolExecutor(max_workers=len(target_models)) as executor:
                future_to_model = {executor.submit(run_model_solver_async, m, file_path): m for m in target_models}
                
                for future in future_to_model:
                    json_result = future.result()
                    if json_result and os.path.exists(json_result):
                        generated_jsons.append(json_result)
            
            # 4. 분석 실행
            if generated_jsons:
                safe_print("\n[Master] 연산 완료. 자동 분석 스크립트를 실행합니다...")
                subprocess.run(["python", "analyze_model_results.py"] + generated_jsons)
                
    except KeyboardInterrupt:
        pass
    finally:
        safe_print("\n[Master] 시스템을 종료합니다.")

if __name__ == "__main__":
    main()