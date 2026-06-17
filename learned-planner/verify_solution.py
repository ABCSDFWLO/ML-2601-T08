import json
import sys
import os

# 소코반 맵의 기호 정의
WALL = '#'
TARGET = '.'
BOX = '$'
BOX_ON_TARGET = '*'
PLAYER = '@'
PLAYER_ON_TARGET = '+'

def parse_txt_map(filepath):
    """원본 txt 파일에서 맵 데이터를 파싱하여 딕셔너리로 반환"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    maps = {}
    current_map = []
    map_id = None
    
    for line in content.split('\n'):
        if line.startswith(';'):
            if current_map and map_id is not None:
                maps[map_id] = current_map
            # [수정된 부분] 맵 번호를 추출한 뒤, JSON과 매칭되도록 3자리(000)로 패딩
            raw_id = line.replace(';', '').strip()
            map_id = raw_id.zfill(3) 
            current_map = []
        elif line.strip():
            current_map.append(line)
            
    if current_map and map_id is not None:
        maps[map_id] = current_map
        
    return maps

def extract_initial_state(map_lines):
    """문자열 맵에서 벽, 목표물, 상자, 플레이어의 좌표를 추출"""
    walls = set()
    targets = set()
    boxes = set()
    player = None
    
    for r, line in enumerate(map_lines):
        for c, char in enumerate(line):
            pos = (r, c)
            if char == WALL: walls.add(pos)
            elif char == TARGET: targets.add(pos)
            elif char == BOX: boxes.add(pos)
            elif char == BOX_ON_TARGET:
                boxes.add(pos)
                targets.add(pos)
            elif char == PLAYER: player = pos
            elif char == PLAYER_ON_TARGET:
                player = pos
                targets.add(pos)
                
    return walls, targets, boxes, player

def simulate_sokoban(walls, targets, boxes, player, action_sequence):
    """행동 시퀀스를 적용하여 맵을 물리적으로 시뮬레이션"""
    directions = {
        'U': (-1, 0), 'D': (1, 0),
        'L': (0, -1), 'R': (0, 1)
    }
    
    for action in action_sequence:
        if action not in directions:
            continue
            
        dr, dc = directions[action]
        next_r, next_c = player[0] + dr, player[1] + dc
        next_pos = (next_r, next_c)
        
        # 1. 이동할 곳이 벽인 경우 (이동 무시)
        if next_pos in walls:
            continue
            
        # 2. 이동할 곳에 상자가 있는 경우 (푸시 판정)
        if next_pos in boxes:
            box_next_r = next_r + dr
            box_next_c = next_c + dc
            box_next_pos = (box_next_r, box_next_c)
            
            # 상자가 밀려날 곳이 벽이거나 또 다른 상자라면 밀 수 없음 (이동 무시)
            if box_next_pos in walls or box_next_pos in boxes:
                continue
                
            # 상자 이동
            boxes.remove(next_pos)
            boxes.add(box_next_pos)
            # 플레이어 이동
            player = next_pos
            
        # 3. 빈 공간이거나 목표물인 경우
        else:
            player = next_pos
            
    # 최종 상태 검증: 모든 목표물 위에 상자가 있는지 확인
    return boxes == targets

def main(txt_file, json_file):
    if not os.path.exists(txt_file) or not os.path.exists(json_file):
        print("에러: 텍스트 파일 또는 JSON 파일을 찾을 수 없습니다.")
        return

    # 데이터 로드
    txt_maps = parse_txt_map(txt_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f).get("data", {})

    print(f"[{os.path.basename(txt_file)}] 독립 검증 시작\n" + "-"*40)
    
    total = 0
    verified_success = 0
    false_positives = 0
    
    for map_key, info in json_data.items():
        # map_key 예시: 'testsokoban.txt_map_000' -> map_id는 '000'
        try:
            map_id = map_key.split('_map_')[-1]
        except IndexError:
            continue
            
        if map_id not in txt_maps:
            print(f"[경고] JSON에는 존재하나 원본 텍스트에 없는 맵: {map_id}")
            continue
            
        total += 1
        ai_claimed_success = (info.get("status") == "success")
        
        # 행동 시퀀스 추출
        actions = [step.get("action") for step in info.get("solution", []) if "action" in step]
        
        # 물리 엔진 세팅 및 시뮬레이션
        walls, targets, boxes, player = extract_initial_state(txt_maps[map_id])
        actual_success = simulate_sokoban(walls, targets, boxes, player, actions)
        
        # 결과 대조
        if actual_success:
            verified_success += 1
            if not ai_claimed_success:
                print(f"[교차오류] 맵 {map_id}: AI는 실패했다고 보고했으나, 실제로는 정답임.")
        else:
            if ai_claimed_success:
                false_positives += 1
                print(f"[치명적 오류/환각] 맵 {map_id}: AI는 성공했다고 보고했으나, 물리 엔진 검증 결과 오답임!")

    print("-" * 40)
    print(f"총 검증 대상 맵 : {total}")
    print(f"실제 정답 처리됨  : {verified_success}")
    print(f"AI 환각(거짓 성공): {false_positives}")
    
    if false_positives == 0:
        print("결론: AI의 해답이 모두 물리적으로 정확함이 검증되었습니다.")
    else:
        print("결론: AI의 해답에 심각한 오류(False Positive)가 포함되어 있습니다.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python verify_solution.py <원본_맵.txt> <결과물.json>")
    else:
        main(sys.argv[1], sys.argv[2])