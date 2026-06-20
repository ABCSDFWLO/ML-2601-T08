import json
import sys
import os

def analyze_models(file_paths):
    models_data = {}
    all_maps = set()
    
    # 1. JSON 데이터 병합 및 무결성 확인
    for path in file_paths:
        if not os.path.exists(path):
            print(f"[오류] 파일을 찾을 수 없습니다: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[오류] 유효하지 않은 JSON 형식입니다: {path}")
                continue
                
            model_name = data.get("model", os.path.basename(path))
            models_data[model_name] = data.get("data", {})
            all_maps.update(models_data[model_name].keys())
            
    if not models_data:
        return

    results = {}
    
    # 2. 모델별 교차 지표 계산
    for target_model, target_data in models_data.items():
        solved_maps = {m for m, d in target_data.items() if d.get("status") == "success"}
        total_maps = len(target_data)
        
        # 독점 해결 문제(Unique Solves) 추출
        unique_solves = 0
        for map_id in solved_maps:
            solved_by_others = False
            for other_model, other_data in models_data.items():
                if target_model == other_model:
                    continue
                if other_data.get(map_id, {}).get("status") == "success":
                    solved_by_others = True
                    break
            
            if not solved_by_others:
                unique_solves += 1
                
        # 성공한 맵 기준 효율성 지표 평균 계산
        if solved_maps:
            avg_steps = sum(target_data[m].get("steps", 0) for m in solved_maps) / len(solved_maps)
            avg_time = sum(target_data[m].get("inference_time_ms", 0) for m in solved_maps) / len(solved_maps)
        else:
            avg_steps = 0
            avg_time = 0

        results[target_model] = {
            "total": total_maps,
            "solved": len(solved_maps),
            "acc": (len(solved_maps) / total_maps * 100) if total_maps > 0 else 0,
            "unique": unique_solves,
            "avg_steps": avg_steps,
            "avg_time_ms": avg_time
        }
        
    # 3. 콘솔 테이블 출력
    print("-" * 80)
    print(f"{'Model Name':<15} | {'Acc (%)':<8} | {'Solved':<8} | {'Unique':<8} | {'Avg Steps':<10} | {'Avg Time(ms)'}")
    print("-" * 80)
    for model, res in results.items():
        print(f"{model:<15} | {res['acc']:<8.2f} | {res['solved']:<8} | {res['unique']:<8} | {res['avg_steps']:<10.2f} | {res['avg_time_ms']:.2f}")
    print("-" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python analyze_results.py <경로1.json> <경로2.json> ...")
    else:
        analyze_models(sys.argv[1:])