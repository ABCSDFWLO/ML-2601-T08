import os
import json
import glob

def update_json_files(directory):
    json_files = glob.glob(os.path.join(directory, "*.json"))
    
    if not json_files:
        print("[ERROR] 해당 경로에서 JSON 파일을 찾을 수 없습니다.")
        return

    for file_path in json_files:
        filename = os.path.basename(file_path)

        # 1. 자동 파싱 제거: 무조건 사용자에게 맵 파일명 입력 요청
        print(f"\n========================================")
        map_filename = input(f"[INPUT] 대상 파일: '{filename}'\n적용할 원본 맵 파일명을 입력하세요 (예: medium_000.txt / 건너뛰려면 Enter): ").strip()
        
        if not map_filename:
            print(f"[SKIP] {filename} 수정을 건너뜁니다.")
            continue

        # 2. JSON 읽기
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"[ERROR] JSON 파싱 실패 (손상된 파일): {filename}")
                continue

        updated = False
        
        # 3. 최상단 "model": "thinker" 추가 (맨 위로 강제 정렬)
        if data.get("model") != "thinker":
            new_root = {"model": "thinker"}
            new_root.update(data)
            data = new_root
            updated = True

        # 4. "data" 내부의 맵 키 이름 변경
        if "data" in data:
            new_data_dict = {}
            for key, value in data["data"].items():
                if key.startswith("map_"):
                    # 기본 상태 (map_000)
                    new_key = f"{map_filename}_{key}"
                    new_data_dict[new_key] = value
                    updated = True
                elif not key.startswith(map_filename): 
                    # 이전 실행으로 잘못된 접두사가 붙어있는 경우 강제 교체 (예: medium.txt_map_000 -> medium_000.txt_map_000)
                    if "_map_" in key:
                        correct_key = f"{map_filename}_map_{key.split('_map_')[1]}"
                        new_data_dict[correct_key] = value
                        updated = True
                    else:
                        new_data_dict[key] = value
                else:
                    new_data_dict[key] = value
            
            if updated:
                data["data"] = new_data_dict

        # 5. 변경된 결과 덮어쓰기
        if updated:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"[SUCCESS] 수정 완료 (적용된 맵 접두사: {map_filename})")
        else:
            print(f"[PASS] 변경 대상 없음 (이미 최신 상태)")

if __name__ == "__main__":
    target_dir = input("수정할 JSON 파일들이 위치한 디렉토리 경로를 입력하세요 (현재 폴더면 . 입력): ").strip()
    update_json_files(target_dir)