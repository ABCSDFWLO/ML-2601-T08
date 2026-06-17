# 파일명: drc_solver.py (윈도우 호스트용)
import os
import time
import subprocess
from datetime import datetime

# [설정 변수]
CONTAINER_NAME = "drc_solver_env"
INTERNAL_WORKSPACE = "/workspace"
INTERNAL_INBOX = f"{INTERNAL_WORKSPACE}/inbox"
INTERNAL_TASK_FILE = f"{INTERNAL_WORKSPACE}/task.txt"
INTERNAL_OUTPUT_JSON = f"{INTERNAL_WORKSPACE}/DRC33_results.json"

LOCAL_OUTPUT_DIR = os.getcwd()

def log_print(message):
    """현재 시간을 포함하여 터미널에 출력하는 함수"""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[{current_time}] {message}")

def run_cmd(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')

def start_docker_environment():
    print("[Master] Starting Docker container...")
    run_cmd(["docker", "start", CONTAINER_NAME])
    
    # [추가] root 권한으로 캐시 폴더 생성 및 심볼릭 링크 강제 연결
    print("[Master] Applying cache path patch as root...")
    run_cmd(["docker", "exec", "-u", "root", CONTAINER_NAME, "mkdir", "-p", "/workspace/.sokoban_cache"])
    run_cmd(["docker", "exec", "-u", "root", CONTAINER_NAME, "ln", "-sf", "/workspace/.sokoban_cache", "/opt/sokoban_cache"])
    
    print("[Master] Cleaning up previous tasks...")
    run_cmd(["docker", "exec", CONTAINER_NAME, "rm", "-f", INTERNAL_TASK_FILE])
    run_cmd(["docker", "exec", CONTAINER_NAME, "rm", "-rf", INTERNAL_INBOX])
    run_cmd(["docker", "exec", CONTAINER_NAME, "mkdir", "-p", INTERNAL_INBOX])
    
    print("[Master] Starting DRC33 Daemon in background...")
    run_cmd(["docker", "exec", "-d", CONTAINER_NAME, "python", f"{INTERNAL_WORKSPACE}/daemon_solver.py"])
    
    print("[Master] Waiting for DRC33 model to load into VRAM (10s)...")
    time.sleep(10)

def check_docker_task_done():
    res = run_cmd(["docker", "exec", CONTAINER_NAME, "ls", INTERNAL_TASK_FILE])
    return res.returncode != 0

def main():
    start_docker_environment()
    
    while True:
        try:
            win_path = input("\n[Master] 평가할 윈도우 파일/폴더 경로를 입력하세요 (종료: 'q'): ").strip()
            
            if win_path.lower() in ['q', 'quit', 'exit']:
                with open("temp_exit.txt", "w") as f: f.write("exit")
                run_cmd(["docker", "cp", "temp_exit.txt", f"{CONTAINER_NAME}:{INTERNAL_TASK_FILE}"])
                os.remove("temp_exit.txt")
                print("[Master] Shutting down...")
                break
            
            win_path = os.path.abspath(win_path)
                
            if not os.path.exists(win_path):
                print(f"[Error] 윈도우 경로를 찾을 수 없습니다: {win_path}")
                continue

            base_name = os.path.basename(win_path.rstrip("\\/"))
            internal_target = f"{INTERNAL_INBOX}/{base_name}"

            print(f"[Master] Copying data to Docker ({base_name})...")
            run_cmd(["docker", "exec", CONTAINER_NAME, "rm", "-rf", INTERNAL_INBOX])
            run_cmd(["docker", "exec", CONTAINER_NAME, "mkdir", "-p", INTERNAL_INBOX])
            
            cp_res = run_cmd(["docker", "cp", win_path, f"{CONTAINER_NAME}:{INTERNAL_INBOX}/"])
            if cp_res.returncode != 0:
                print(f"[Error] 복사 실패: {cp_res.stderr}")
                continue

            print("[Master] Triggering Docker daemon...")
            with open("temp_task.txt", "w") as f: f.write(internal_target)
            run_cmd(["docker", "cp", "temp_task.txt", f"{CONTAINER_NAME}:{INTERNAL_TASK_FILE}"])
            os.remove("temp_task.txt")

            log_print("[Master] Waiting for Docker to finish...")
            while not check_docker_task_done():
                time.sleep(1)

            log_print("[Master] Retrieving results from Docker...")
            local_json_path = os.path.join(LOCAL_OUTPUT_DIR, f"DRC33_{base_name}_results.json")
            run_cmd(["docker", "cp", f"{CONTAINER_NAME}:{INTERNAL_OUTPUT_JSON}", local_json_path])
            log_print(f"[Master] Done! Saved locally to: {local_json_path}")

        except KeyboardInterrupt:
            with open("temp_exit.txt", "w") as f: f.write("exit")
            run_cmd(["docker", "cp", "temp_exit.txt", f"{CONTAINER_NAME}:{INTERNAL_TASK_FILE}"])
            if os.path.exists("temp_exit.txt"): os.remove("temp_exit.txt")
            break

if __name__ == "__main__":
    main()