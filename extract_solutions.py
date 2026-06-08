# python .\extract_solutions.py로 실행
# Ctrl + C 누를 경우 저장 후 종료됨

import os
import shutil
import subprocess
import time
import json
import re
import queue
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed

# [설정]
MAP_DIR = os.path.join(os.path.dirname(__file__), "boxoban-levels", "unfiltered", "train")
FESTIVAL_BIN_DIR = os.path.join(os.path.dirname(__file__), "FESTIVAL_SOLVER", "bin")
SOLVER_PATH = os.path.join(FESTIVAL_BIN_DIR, "festival.exe")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "solutions.json")
WORKER_DIR_BASE = os.path.join(os.path.dirname(__file__), "workers")

TIMEOUT_SECONDS = 15.0
CHECKPOINT_INTERVAL = 500
MAX_WORKERS = 2

def get_hardware_info():
    hardware_info = {
        "OS": f"{platform.system()} {platform.release()}",
        "CPU": platform.processor(),
        "Logical_Cores": os.cpu_count(),
        "GPU": "Not Detected"
    }
    return hardware_info

def setup_workers(max_workers):
    os.makedirs(WORKER_DIR_BASE, exist_ok=True)
    worker_q = queue.Queue()
    for i in range(max_workers):
        wdir = os.path.join(WORKER_DIR_BASE, f"worker_{i}")
        os.makedirs(wdir, exist_ok=True)
        shutil.copy2(SOLVER_PATH, wdir)
        dll_path = os.path.join(FESTIVAL_BIN_DIR, "libwinpthread-1.dll")
        if os.path.exists(dll_path):
            shutil.copy2(dll_path, wdir)
        for f in ["solutions.sok", "times.txt", "map.sok", "debug.log"]:
            fp = os.path.join(wdir, f)
            if os.path.exists(fp):
                try: os.remove(fp)
                except OSError: pass
        worker_q.put(wdir)
    return worker_q

def solve_single_map(map_text, map_id, worker_queue):
    wdir = worker_queue.get()
    start_time = time.perf_counter()
    status = "timeout"
    solution = None
    debug_log = ""
    solve_time_ms = TIMEOUT_SECONDS * 1000.0

    map_path = os.path.join(wdir, "map.sok")
    sol_file = os.path.join(wdir, "solutions.sok")
    log_file = os.path.join(wdir, "debug.log")
    exe_path = os.path.join(wdir, "festival.exe")

    for f in [sol_file, log_file, os.path.join(wdir, "times.txt")]:
        if os.path.exists(f):
            try: os.remove(f)
            except OSError: pass

    with open(map_path, 'w', encoding='utf-8') as f:
        f.write(f"; {map_id}\n; 0\n{map_text}\n")

    try:
        CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
        with open(log_file, 'wb') as log_out:
            subprocess.run(
                [exe_path, "map.sok", "-time", str(int(TIMEOUT_SECONDS))],
                stdout=log_out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                cwd=wdir, creationflags=CREATE_NO_WINDOW, timeout=TIMEOUT_SECONDS + 2.0
            )
        status = "processing_complete"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except Exception as e:
        status = "error"
        debug_log = f"System Error: {str(e)}"

    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as lf:
            debug_log = lf.read().strip()

    if os.path.exists(sol_file):
        with open(sol_file, 'r', encoding='utf-8', errors='ignore') as sf:
            for line in sf.read().splitlines():
                line = line.strip()
                if line and all(c in 'uUdDlLrR' for c in line):
                    solution = line
                    status = "success"
                    break

    if status == "success":
        solve_time_ms = (time.perf_counter() - start_time) * 1000.0

    if status != "success" and status != "error":
        if "unsolvable" in debug_log.lower() or "too many moves" in debug_log.lower():
            status = "unsolvable_or_rejected"
        elif not debug_log:
            debug_log = "No output from solver. File empty."

    worker_queue.put(wdir)
    return map_id, solution, solve_time_ms, status, debug_log

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return [m.strip() for m in re.split(r'\n\s*\n', content) if m.strip()]

def save_json(output_path, hw_info, total_processed, stats, results, elapsed_time):
    output_data = {
        "metadata": {
            "hardware": hw_info,
            "performance": {
                "total_maps_processed_this_session": total_processed,
                "success_count_this_session": stats["success"],
                "failed_count_this_session": stats.get("timeout", 0) + stats.get("timeout_or_unsolvable", 0) + stats.get("error", 0) + stats.get("unsolvable_or_rejected", 0),
                "session_wall_clock_time_seconds": round(elapsed_time, 4),
                "total_maps_in_database": len(results)
            }
        },
        "data": results
    }
    tmp_path = output_path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)
    os.replace(tmp_path, output_path)

def build_solutions_json():
    if not os.path.exists(SOLVER_PATH):
        print(f"[Error] 원본 솔버를 찾을 수 없습니다: {SOLVER_PATH}")
        return

    hw_info = get_hardware_info()
    worker_queue = setup_workers(MAX_WORKERS)

    results = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                results = existing_data.get("data", {})
            print(f"[System] 기존 데이터 {len(results)}개 로드 완료. 이어하기를 시작합니다.")
        except json.JSONDecodeError:
            pass

    map_files = [f for f in os.listdir(MAP_DIR) if f.endswith('.txt')]
    all_individual_maps = []
    
    for file_name in map_files:
        file_path = os.path.join(MAP_DIR, file_name)
        extracted_maps = process_file(file_path)
        for idx, map_text in enumerate(extracted_maps):
            map_id = f"{file_name}_map_{idx+1:03d}"
            if map_id not in results:
                all_individual_maps.append((map_id, map_text))

    total_maps_to_run = len(all_individual_maps)
    if total_maps_to_run == 0:
        print("[System] 모든 맵의 처리가 완료되었습니다.")
        return

    print(f"[System] 대상 맵: {total_maps_to_run}개 (동시 워커 {MAX_WORKERS}개)")
    
    stats = {"success": 0, "timeout": 0, "timeout_or_unsolvable": 0, "error": 0}
    processed_count = 0
    start_wall_time = time.time()
    
    # 핵심 변경점: with 구문(종료 시 스레드 대기) 폐기
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    try:
        future_to_map = {
            executor.submit(solve_single_map, map_text, map_id, worker_queue): map_id 
            for map_id, map_text in all_individual_maps
        }

        for future in as_completed(future_to_map):
            map_id, solution, solve_time_ms, status, debug_log = future.result()
            
            if status == "timeout_or_unsolvable" or status == "processing_complete":
                status = "timeout_or_unsolvable"
                
            stats[status] += 1
            processed_count += 1
            
            results[map_id] = {
                "status": status,
                "solve_time_ms": round(solve_time_ms, 2),
                "steps": len(solution) if solution else 0,
                "solution": solution
            }
            
            if status != "success":
                results[map_id]["debug_stdout"] = debug_log[-800:] if debug_log else "No output from solver"

            if processed_count % 10 == 0:
                print(f"진행: {processed_count}/{total_maps_to_run} 완료 (성공: {stats['success']})")

            if processed_count % CHECKPOINT_INTERVAL == 0:
                save_json(OUTPUT_JSON, hw_info, processed_count, stats, results, time.time() - start_wall_time)
                print(f"[Checkpoint] {processed_count}개 처리 및 디스크 저장 완료.")

        # 루프가 정상적으로 끝났을 때의 처리
        save_json(OUTPUT_JSON, hw_info, processed_count, stats, results, time.time() - start_wall_time)
        executor.shutdown(wait=True)
        print(f"\n[Success] JSON 저장 완료. 경로: {OUTPUT_JSON}")

    except KeyboardInterrupt:
        # 하드 킬(Hard Kill) 시퀀스 가동
        print("\n\n[System] Ctrl+C 감지! 강제 종료 시퀀스 가동 중...")
        
        # 1. 실행 대기 중인 모든 큐 취소 (스레드 추가 할당 차단)
        executor.shutdown(wait=False, cancel_futures=True)
        
        # 2. 현재까지 메모리에 있는 데이터를 무조건 디스크에 기록
        print("[System] 현재까지 처리된 데이터를 안전하게 저장합니다...")
        save_json(OUTPUT_JSON, hw_info, processed_count, stats, results, time.time() - start_wall_time)
        
        # 3. 백그라운드에서 CPU를 점유 중인 솔버 즉각 사살
        print("[System] 백그라운드 연산 프로세스(festival.exe) 강제 정리 중...")
        if platform.system() == "Windows":
            subprocess.run("taskkill /F /IM festival.exe /T", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        # 4. 데드락 무시하고 파이썬 프로세스 메모리 즉각 반환 (빠른 종료)
        print("[System] 프로그램이 안전하게 종료되었습니다.")
        os._exit(0)

if __name__ == "__main__":
    build_solutions_json()