import json
import sys
import os

def analyze_models(file_paths):
    models_data = {}
    
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
            
    if not models_data:
        return

    results = {}
    all_map_ids = set()
    all_solved_sets = {}
    
    # 2. 모든 맵 ID와 성공한 맵 ID 추출 (합집합 연산용)
    for model, data in models_data.items():
        solved_ids = set()
        for key, val in data.items():
            try:
                map_id = key.split('_map_')[-1]
            except IndexError:
                continue
                
            all_map_ids.add(map_id)
            if val.get("status") == "success":
                solved_ids.add(map_id)
                
        all_solved_sets[model] = solved_ids
    
    # 3. 모델별 교차 지표 및 효율성 계산
    for target_model, target_data in models_data.items():
        target_solved_ids = all_solved_sets[target_model]
        total_maps = len(target_data)
        
        # 차집합 연산을 통한 독점 해결 문제 추출
        others_solved_ids = set().union(*(all_solved_sets[m] for m in all_solved_sets if m != target_model))
        unique_solves = len(target_solved_ids - others_solved_ids)
                
        # 효율성 지표 평균 계산
        solved_original_keys = [k for k, v in target_data.items() if v.get("status") == "success"]
        
        if solved_original_keys:
            avg_steps = sum(target_data[k].get("steps", 0) for k in solved_original_keys) / len(solved_original_keys)
            avg_time = sum(target_data[k].get("inference_time_ms", 0) for k in solved_original_keys) / len(solved_original_keys)
        else:
            avg_steps = 0
            avg_time = 0

        results[target_model] = {
            "total": total_maps,
            "solved": len(target_solved_ids),
            "acc": (len(target_solved_ids) / total_maps * 100) if total_maps > 0 else 0,
            "unique": unique_solves,
            "avg_steps": avg_steps,
            "avg_time_ms": avg_time
        }
        
    # 4. 콘솔 테이블 및 통합 결과 출력
    print("-" * 80)
    print(f"{'Model Name':<15} | {'Acc (%)':<8} | {'Solved':<8} | {'Unique':<8} | {'Avg Steps':<10} | {'Avg Time(ms)'}")
    print("-" * 80)
    for model, res in results.items():
        print(f"{model:<15} | {res['acc']:<8.2f} | {res['solved']:<8} | {res['unique']:<8} | {res['avg_steps']:<10.2f} | {res['avg_time_ms']:.2f}")
    print("-" * 80)
    
    # [추가됨] 전체 모델 통합 해결 지표 계산
    total_solved_union = len(set().union(*all_solved_sets.values()))
    total_unique_maps = len(all_map_ids)
    print(f"Total Combined Solved: [{total_solved_union}/{total_unique_maps}]")
    print("-" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python analyze_results.py <경로1.json> <경로2.json> ...")
    else:
        analyze_models(sys.argv[1:])