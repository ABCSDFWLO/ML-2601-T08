import sys
import os
import time
import json
import subprocess

CONTAINER_NAME = "thinker_solver_env"
INTERNAL_WORKSPACE = "/workspace"
INTERNAL_INBOX = f"{INTERNAL_WORKSPACE}/inbox"
INTERNAL_TASK_FILE = f"{INTERNAL_WORKSPACE}/task.txt"
INTERNAL_OUTPUT_JSON = f"{INTERNAL_WORKSPACE}/THINKER_results.json"

LOCAL_OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "solutions"))

def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')

def evaluate_in_docker(win_path):
    model_name = "thinker"  # 소문자로 통일
    rec_t = 20              # 규격 통일용 파라미터
    
    win_path = os.path.abspath(win_path)
    if not os.path.exists(win_path):
        print(f"[{model_name}] 파일 누락: {win_path}", flush=True)
        return

    base_name = os.path.basename(win_path)
    internal_target = f"{INTERNAL_INBOX}/{base_name}"
    
    # thinker_testsokoban.txt_20_results.json 형식으로 저장
    local_json_path = os.path.join(LOCAL_OUTPUT_DIR, f"{model_name}_{base_name}_{rec_t}_results.json")

    # [수정 1] 이전 연산의 잔재로 인한 Race Condition 방지 (Output 강제 삭제 추가)
    run_cmd(["docker", "exec", CONTAINER_NAME, "rm", "-f", INTERNAL_TASK_FILE])
    run_cmd(["docker", "exec", CONTAINER_NAME, "rm", "-f", INTERNAL_OUTPUT_JSON])
    run_cmd(["docker", "exec", CONTAINER_NAME, "rm", "-rf", INTERNAL_INBOX])
    run_cmd(["docker", "exec", CONTAINER_NAME, "mkdir", "-p", INTERNAL_INBOX])
    
    run_cmd(["docker", "cp", win_path, f"{CONTAINER_NAME}:{internal_target}"])
    run_cmd(["docker", "exec", CONTAINER_NAME, "sh", "-c", f"echo '{internal_target}' > {INTERNAL_TASK_FILE}"])
    
    print(f"[{model_name}] 도커 내부로 파일 복사 완료 ({base_name})", flush=True)
    print(f"[{model_name}] 추론 연산 진행 중... (대기)", flush=True)

    while True:
        inspect_res = run_cmd(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME])
        if inspect_res.stdout.strip() != "true":
            print(f"[{model_name}] 치명적 오류: 도커 컨테이너가 예기치 않게 종료되었습니다.", flush=True)
            return

        res = run_cmd(["docker", "exec", CONTAINER_NAME, "ls", INTERNAL_OUTPUT_JSON])
        if res.returncode == 0:
            break
        time.sleep(0.5)

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

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
    if len(sys.argv) < 2:
        sys.exit(1)
    evaluate_in_docker(sys.argv[1])