import sys
import os
import time
import json
import subprocess

CONTAINER_NAME = "drc_solver_env"
INTERNAL_WORKSPACE = "/workspace"
INTERNAL_INBOX = f"{INTERNAL_WORKSPACE}/inbox"
INTERNAL_TASK_FILE = f"{INTERNAL_WORKSPACE}/task.txt"
INTERNAL_OUTPUT_JSON = f"{INTERNAL_WORKSPACE}/DRC33_results.json"

# 루트 디렉터리 하위의 solutions 폴더로 경로 변경
LOCAL_OUTPUT_DIR = os.path.join(os.getcwd(), "solutions")

def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')

def evaluate_in_docker(win_path):
    model_name = "DRC33"
    win_path = os.path.abspath(win_path)
    if not os.path.exists(win_path):
        print(f"[{model_name}] 파일 누락: {win_path}")
        return

    base_name = os.path.basename(win_path)
    internal_target = f"{INTERNAL_INBOX}/{base_name}"
    local_json_path = os.path.join(LOCAL_OUTPUT_DIR, f"{model_name}_{base_name}_results.json")

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

    # 1. 이전 작업 잔재 완전 초기화 (sh -c 사용)
    run_cmd(["docker", "exec", CONTAINER_NAME, "sh", "-c", f"rm -f {INTERNAL_TASK_FILE} {INTERNAL_OUTPUT_JSON}"])
    run_cmd(["docker", "exec", CONTAINER_NAME, "sh", "-c", f"rm -rf {INTERNAL_INBOX}/*"])
    run_cmd(["docker", "exec", CONTAINER_NAME, "mkdir", "-p", INTERNAL_INBOX])
    
    # 2. 파일 복사
    cp_res = run_cmd(["docker", "cp", win_path, f"{CONTAINER_NAME}:{INTERNAL_INBOX}/"])
    if cp_res.returncode != 0:
        print(f"[{model_name}] 복사 실패: {cp_res.stderr}")
        return

    print(f"[{model_name}] 도커 내부로 파일 복사 완료 ({base_name})", flush=True)

    # 3. 데몬에 작업 지시
    with open("temp_task.txt", "w", encoding='utf-8') as f: f.write(internal_target)
    run_cmd(["docker", "cp", "temp_task.txt", f"{CONTAINER_NAME}:{INTERNAL_TASK_FILE}"])
    os.remove("temp_task.txt")

    print(f"[{model_name}] 추론 연산 진행 중... (대기)", flush=True)

    # 4. 결과 대기 및 상태 검증
    while True:
        inspect_res = run_cmd(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME])
        if inspect_res.stdout.strip() != "true":
            print(f"[{model_name}] 치명적 오류: 도커 컨테이너가 예기치 않게 종료되었습니다.")
            return

        res = run_cmd(["docker", "exec", CONTAINER_NAME, "ls", INTERNAL_OUTPUT_JSON])
        if res.returncode == 0:
            break
        time.sleep(0.5)

    # 5. 결과 반출 및 파싱
    run_cmd(["docker", "cp", f"{CONTAINER_NAME}:{INTERNAL_OUTPUT_JSON}", local_json_path])
    run_cmd(["docker", "exec", CONTAINER_NAME, "rm", "-f", INTERNAL_OUTPUT_JSON])

    try:
        with open(local_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # JSON에 기록된 모든 맵 데이터의 연산 시간을 합산
        total_solve_time = sum(val.get("total_system_time_ms", 0.0) for val in data.get("data", {}).values())
        
        # 합산된 총 소요시간 출력
        print(f"[{model_name}] 완료 | 총 소요시간: {total_solve_time:.2f}ms | 파일: {base_name}", flush=True)
        
        # 메인 라우터로 저장된 파일 경로 전달 (화면에는 숨김 처리됨)
        print(f"[JSON_OUTPUT] {local_json_path}", flush=True)
        
    except Exception as e:
        print(f"[{model_name}] 결과 파싱 실패 ({base_name}): {e}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        evaluate_in_docker(sys.argv[1])
    else:
        print("[DRC33] 입력 인자가 부족합니다.")