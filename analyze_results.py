import json
import sys

def main(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"에러: 파일을 읽는 중 문제가 발생했습니다 - {e}")
        return

    results = json_data.get("data", {})
    total = len(results)

    if total == 0:
        print("에러: 파싱할 맵 데이터가 없습니다.")
        return

    solved_count = 0
    total_inference_ms = 0.0
    total_system_ms = 0.0

    for map_name, info in results.items():
        if info.get("status") == "success":
            solved_count += 1
        
        # 두 가지 시간 지표 모두 누적 (이전 데이터 호환성을 위해 기본값 0.0 처리)
        total_inference_ms += info.get("inference_time_ms", info.get("solve_time_ms", 0.0))
        total_system_ms += info.get("total_system_time_ms", info.get("solve_time_ms", 0.0))

    unsolved_count = total - solved_count
    avg_inference_ms = total_inference_ms / total
    avg_system_ms = total_system_ms / total

    print(f"=== 분석 결과 ===")
    print(f"대상 파일      : {file_path}")
    print(f"총 문제 수     : {total}")
    print(f"푼 문제 수     : {solved_count}")
    print(f"풀지 못한 문제 : {unsolved_count}")
    print(f"해결 비율      : {(solved_count / total) * 100:.2f}%\n")
    print(f"[소요 시간 통계]")
    print(f"평균 순수 추론 시간 (모델) : {avg_inference_ms:.2f} ms")
    print(f"평균 총 소요 시간 (시스템) : {avg_system_ms:.2f} ms")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python analyze_results.py <json_파일_경로>")
    else:
        main(sys.argv[1])