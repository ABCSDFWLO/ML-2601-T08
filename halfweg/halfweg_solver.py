import argparse
import json
import os
import types
import sys
import time
import contextlib 
from pathlib import Path

import numpy as np
import torch


# ROOT_DIR은 'halfweg' 폴더를 가리킴
ROOT_DIR = Path(__file__).resolve().parent
HALFWEG_SRC = ROOT_DIR / "src"
if str(HALFWEG_SRC) not in sys.path:
    sys.path.insert(0, str(HALFWEG_SRC))

# halfweg/src/helpers.py imports Unix-only `resource`; provide a small shim on Windows.
if "resource" not in sys.modules:
    shim = types.ModuleType("resource")
    shim.RUSAGE_SELF = 0

    class _RUsage:
        ru_maxrss = 0

        def __getitem__(self, index):
            return self.ru_maxrss if index == 2 else 0

    def _getrusage(_who):
        return _RUsage()

    shim.getrusage = _getrusage
    sys.modules["resource"] = shim

from environments.sokoban.sokoban_env import Sokoban
from environments.sokoban.sokoban_levels_manager import board_2d_to_3d, load_file_all_boards, split_to_level_boards
from hw_impl import env_torch_wrapper, evaluate, hw_common, hw_experience_replay, model_mgmt


ACTION_TO_CHAR = {
    0: "U",
    1: "R",
    2: "D",
    3: "L",
}


def _guess_model_name(ckpt_path: str) -> str:
    stem = Path(ckpt_path).stem
    if stem.lower().startswith("checkpoint_"):
        return "HalfWeg"
    return stem.split("_")[0] if "_" in stem else stem


def _build_config_for_checkpoint(ckpt_path: str, plta_class: str, plhw_class: str) -> dict:
    return {
        "model": {
            "PLTA": {"class": plta_class},
            "PLHW": {"class": plhw_class},
            "checkpoint": ckpt_path,
        }
    }


def _auto_detect_model_classes(ckpt_path: str) -> tuple[str, str]:
    candidates = ["v4_1010", "v3_1010", "v2_1010", "v1_1010"]
    errors = []

    for suffix in candidates:
        plta = f"Sokoban_PLTA_{suffix}"
        plhw = f"Sokoban_PLHW_{suffix}"
        config = _build_config_for_checkpoint(ckpt_path, plta, plhw)
        try:
            _ = model_mgmt.ModelKeeper(config)
            return plta, plhw
        except Exception as exc:
            errors.append(f"{plta}/{plhw}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "Could not infer model classes from checkpoint. "
        "Provide --plta-class and --plhw-class manually.\n"
        f"Tried:\n{joined}"
    )


def _choose_target_idx(curr_state: np.ndarray, targets_np: np.ndarray) -> int:
    curr_boxes = curr_state[2]
    target_boxes = targets_np[:, 2, :, :]
    mismatch = np.mean(np.abs(target_boxes - curr_boxes), axis=(1, 2))
    return int(np.argmin(mismatch))


def _pad_board_to_shape(board_2d: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
    rows, cols = board_2d.shape
    if rows == target_rows and cols == target_cols:
        return board_2d
    if rows > target_rows or cols > target_cols:
        raise ValueError(
            f"Level board shape {board_2d.shape} exceeds model input {(target_rows, target_cols)}"
        )

    padded = np.zeros((target_rows, target_cols), dtype=board_2d.dtype)
    padded[:rows, :cols] = board_2d
    return padded


@torch.no_grad()
def _solve_one_map(
    env: Sokoban,
    plta,
    policy,
    device,
    max_steps: int,
) -> dict:
    start_t = time.perf_counter()

    targets_np = env.get_target_states()
    targets_t = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
    targets_enc = plta.forward_model_board_normalize(targets_t)

    AS = plta.AS
    b_towards = hw_common.get_b_array_from_towards_or_away(True, cnt=1, device=device)

    solution = []
    inference_ms = 0.0

    for step_idx in range(1, max_steps + 1):
        if env.done:
            break

        t0 = time.perf_counter()

        curr_state = env.get_model_input_s()
        target_i = _choose_target_idx(curr_state, targets_np)

        curr_envs = env_torch_wrapper.EnvsTensorList(envs=[env.copy()])
        target_envs = env_torch_wrapper.EnvsTensorList(states_t=targets_t[target_i : target_i + 1])

        plan_t = policy.get_plan_envs_to_envs(s0=curr_envs, target=target_envs, b=b_towards)
        plan_np = plan_t[0].detach().cpu().numpy()

        logits_t = plta.forward_model_target_actions__enc(
            ss=plta.forward_model_board_normalize(curr_envs.get_states_t().to(device)),
            ts=targets_enc[target_i : target_i + 1],
            min_max_01=b_towards,
        )
        step_logits = logits_t[0, 0, :AS].detach().cpu().numpy().astype(float)

        action = None
        valid_mask = np.array(env.get_valid_actions_mask(), dtype=np.int64)

        for a in plan_np:
            a = int(a)
            if a == AS:
                break
            if 0 <= a < AS and valid_mask[a] == 1:
                action = a
                break

        if action is None:
            masked = step_logits.copy()
            masked[valid_mask == 0] = -1e9
            action = int(np.argmax(masked))

        reward, done = env.step(action)

        fwd = time.perf_counter() - t0
        inference_ms += fwd * 1000.0

        # 수정 지점 1: policy_logits 및 value 키 제거
        solution.append(
            {
                "step": step_idx,
                "forward_time": float(fwd),
                "action": ACTION_TO_CHAR.get(action, str(action)),
            }
        )

        if reward == 1 or done:
            break

    solved = env.all_boxes_correct()
    total_ms = (time.perf_counter() - start_t) * 1000.0

    return {
        "status": "success" if solved else "failed",
        "steps": len(solution),
        "inference_time_ms": float(inference_ms),
        "total_system_time_ms": float(total_ms),
        "solution": solution,
    }


def run_wrapper(args: argparse.Namespace) -> dict:
    ckpt_path = os.path.abspath(args.ckpt)
    txt_path = os.path.abspath(args.txt)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"Level txt not found: {txt_path}")

    if args.device == "cpu":
        device = "cpu"
    elif args.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # [수정된 핵심 로직] 이 블록 내부에서 발생하는 모든 내부 라이브러리의 print 출력을 강제 차단합니다.
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            if args.plta_class and args.plhw_class:
                plta_class = args.plta_class
                plhw_class = args.plhw_class
            else:
                plta_class, plhw_class = _auto_detect_model_classes(ckpt_path)

            config = _build_config_for_checkpoint(ckpt_path, plta_class, plhw_class)

            model_keeper = model_mgmt.ModelKeeper(config)
            model_keeper.to(device)
            model_keeper.eval()

            envs_sampler = hw_experience_replay.MemoryEnvsSampler(model_keeper=model_keeper)
            policy_name = args.policy
            if policy_name == "last":
                policy_name = evaluate.get_policies(model_keeper, "last")[0]
            policy = evaluate.get_policy(policy_name, model_keeper, envs_sampler)
            policy.eval()

            plta = model_keeper.models["PLTA"]
            plta.eval()

    all_boards = load_file_all_boards(txt_path)
    basename = os.path.basename(txt_path)

    data = {}
    target_rows, target_cols = plta.INPUT_SIZE[1], plta.INPUT_SIZE[2]
    
    for i, board_2d in enumerate(split_to_level_boards(txt_path, all_boards)):
        map_key = f"{basename}_map_{i:03d}"
        board_2d = _pad_board_to_shape(board_2d, target_rows, target_cols)
        env = Sokoban(copy_from=None, init_board=board_2d_to_3d(board_2d))
        data[map_key] = _solve_one_map(
            env=env,
            plta=plta,
            policy=policy,
            device=device,
            max_steps=args.max_steps,
        )

    result = {
        "model": args.model_name if args.model_name else _guess_model_name(ckpt_path),
        "data": data,
    }
    return result


def main() -> None:
    # 1. 단일 인자(맵 파일 경로) 수신 및 검증
    if len(sys.argv) < 2:
        print("[HALFWEG] 입력 인자가 부족합니다. 맵 파일 경로를 전달하십시오.", flush=True)
        sys.exit(1)
        
    txt_path = os.path.abspath(sys.argv[1])
    
    # 2. 콘솔에서 수동으로 입력하던 기존 설정값들을 내부 변수로 고정
    ckpt_path = os.path.join(str(ROOT_DIR), "trained_models", "boxoban_vast_v4_20250421-073320__repacked.ckpt")
    
    # run_wrapper가 요구하는 args 객체를 모방(Mocking)하여 호환성 유지
    class ConfigArgs:
        pass
    
    args = ConfigArgs()
    args.ckpt = ckpt_path
    args.txt = txt_path
    args.output = None
    args.model_name = "halfweg"
    args.max_steps = 200
    args.device = "auto"
    args.policy = "last"
    args.plta_class = None
    args.plhw_class = None

    print(f"[{args.model_name.upper()}] 추론 연산 진행 중... (대기)", flush=True)

    # 3. 모델 추론 실행
    result = run_wrapper(args)

    # 4. 파이프라인 규격에 맞춘 결과 저장 (루트 폴더의 solutions 내부)
    txt_name = os.path.basename(txt_path)
    output_path = os.path.join(str(ROOT_DIR.parent), "solutions", f"halfweg_{txt_name}_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, indent=4, ensure_ascii=False)

    # 5. 최상위 라우터 전달용 통신 태그 출력
    total_solve_time = sum(val.get("total_system_time_ms", 0.0) for val in result.get("data", {}).values())
    
    print(f"[{args.model_name.upper()}] 완료 | 총 소요시간: {total_solve_time:.2f}ms | 파일: {txt_name}", flush=True)
    print(f"[JSON_OUTPUT] {output_path}", flush=True)

if __name__ == "__main__":
    main()