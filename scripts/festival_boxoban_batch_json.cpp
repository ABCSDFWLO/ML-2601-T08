#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

struct SolveResult {
    std::string status;  // success | failed
    double solve_time_ms = 0.0;
    int steps = 0;
    std::string solution;
    std::string fail_reason;
};

struct Task {
    fs::path level_file;
    int level_index_1based = 0;
    std::string key;
};

struct PerformanceBase {
    uint64_t processed = 0;
    uint64_t success = 0;
    uint64_t failed = 0;
    double wall_clock_sec = 0.0;
};

struct LoadState {
    std::unordered_map<std::string, SolveResult> results;
    PerformanceBase perf_base;
};

struct RuntimeCounters {
    std::atomic<uint64_t> newly_processed{0};
    std::atomic<uint64_t> newly_success{0};
    std::atomic<uint64_t> newly_failed{0};
};

std::atomic<bool> g_stop_requested{false};

#ifdef _WIN32
BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT || ctrl_type == CTRL_CLOSE_EVENT) {
        g_stop_requested.store(true);
        return TRUE;
    }
    return FALSE;
}
#endif

void signal_handler(int sig) {
    if (sig == SIGINT) {
        g_stop_requested.store(true);
    }
}

std::string trim_copy(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b])) != 0) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])) != 0) {
        --e;
    }
    return s.substr(b, e - b);
}

std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        switch (c) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u"
                        << std::hex << std::uppercase << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(c))
                        << std::dec << std::nouppercase;
                } else {
                    oss << c;
                }
                break;
        }
    }
    return oss.str();
}

std::string maybe_env(const char* key, const std::string& fallback) {
    const char* value = std::getenv(key);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}

bool is_numeric_stem(const fs::path& p) {
    std::string stem = p.stem().string();
    if (stem.empty()) {
        return false;
    }
    for (char c : stem) {
        if (std::isdigit(static_cast<unsigned char>(c)) == 0) {
            return false;
        }
    }
    return true;
}

bool is_sokoban_char(char c) {
    switch (c) {
        case '#':
        case '@':
        case '$':
        case '.':
        case '*':
        case '+':
        case ' ':
        case '-':
        case '_':
            return true;
        default:
            return false;
    }
}

bool is_sokoban_line(const std::string& raw) {
    std::string s = raw;
    if (!s.empty() && s.back() == '\r') {
        s.pop_back();
    }

    if (s.size() < 3) {
        return false;
    }

    bool has_hash = s.find('#') != std::string::npos;
    if (!has_hash) {
        bool all_space = true;
        for (char c : s) {
            if (std::isspace(static_cast<unsigned char>(c)) == 0) {
                all_space = false;
                break;
            }
        }
        if (!all_space) {
            return false;
        }
    }

    for (char c : s) {
        if (!is_sokoban_char(c) && std::isspace(static_cast<unsigned char>(c)) == 0) {
            return false;
        }
    }
    return true;
}

int count_levels_in_file(const fs::path& file_path) {
    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        return 0;
    }

    int total = 0;
    bool in_level = false;
    std::string line;
    while (std::getline(in, line)) {
        if (is_sokoban_line(line)) {
            in_level = true;
        } else {
            if (in_level) {
                ++total;
            }
            in_level = false;
        }
    }
    if (in_level) {
        ++total;
    }
    return total;
}

std::vector<fs::path> collect_numeric_level_files(const fs::path& root) {
    std::vector<fs::path> files;
    if (!fs::exists(root)) {
        return files;
    }

    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const fs::path& p = entry.path();
        if (p.extension() != ".txt") {
            continue;
        }
        if (!is_numeric_stem(p)) {
            continue;
        }
        files.push_back(fs::absolute(p));
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::vector<Task> build_tasks(const std::vector<fs::path>& files) {
    std::vector<Task> tasks;
    for (const fs::path& f : files) {
        int levels = count_levels_in_file(f);
        std::string name = f.filename().string();
        for (int i = 1; i <= levels; ++i) {
            std::ostringstream key;
            key << name << "_map_" << std::setw(3) << std::setfill('0') << i;
            tasks.push_back(Task{f, i, key.str()});
        }
    }
    return tasks;
}

std::optional<std::string> extract_json_object_after_key(const std::string& json, const std::string& key) {
    const std::string marker = "\"" + key + "\"";
    size_t pos = json.find(marker);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = json.find(':', pos + marker.size());
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    pos = json.find('{', pos + 1);
    if (pos == std::string::npos) {
        return std::nullopt;
    }

    size_t start = pos;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;

    for (size_t i = pos; i < json.size(); ++i) {
        char c = json[i];

        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }

        if (c == '"') {
            in_string = true;
            continue;
        }

        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                return json.substr(start, i - start + 1);
            }
        }
    }

    return std::nullopt;
}

std::optional<std::string> regex_group_1(const std::string& s, const std::regex& re) {
    std::smatch m;
    if (std::regex_search(s, m, re) && m.size() >= 2) {
        return m[1].str();
    }
    return std::nullopt;
}

LoadState load_existing_json(const fs::path& output_path) {
    LoadState state;
    if (!fs::exists(output_path)) {
        return state;
    }

    std::ifstream in(output_path, std::ios::binary);
    if (!in) {
        return state;
    }

    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    {
        std::regex re_processed("\"total_maps_processed_this_session\"\\s*:\\s*([0-9]+)");
        std::regex re_success("\"success_count_this_session\"\\s*:\\s*([0-9]+)");
        std::regex re_failed("\"failed_count_this_session\"\\s*:\\s*([0-9]+)");
        std::regex re_wall("\"session_wall_clock_time_seconds\"\\s*:\\s*([0-9eE+\\-.]+)");

        if (auto g = regex_group_1(json, re_processed)) {
            state.perf_base.processed = static_cast<uint64_t>(std::stoull(*g));
        }
        if (auto g = regex_group_1(json, re_success)) {
            state.perf_base.success = static_cast<uint64_t>(std::stoull(*g));
        }
        if (auto g = regex_group_1(json, re_failed)) {
            state.perf_base.failed = static_cast<uint64_t>(std::stoull(*g));
        }
        if (auto g = regex_group_1(json, re_wall)) {
            state.perf_base.wall_clock_sec = std::stod(*g);
        }
    }

    auto data_obj_opt = extract_json_object_after_key(json, "data");
    if (!data_obj_opt.has_value()) {
        return state;
    }

    const std::string& data_obj = *data_obj_opt;
    size_t i = 1;  // skip opening {

    const std::regex re_status("\"status\"\\s*:\\s*\"([^\"]*)\"");
    const std::regex re_solve_time("\"solve_time_ms\"\\s*:\\s*([0-9eE+\\-.]+)");
    const std::regex re_steps("\"steps\"\\s*:\\s*([0-9]+)");
    const std::regex re_solution("\"solution\"\\s*:\\s*\"([^\"]*)\"");
    const std::regex re_fail_reason("\"fail_reason\"\\s*:\\s*\"([^\"]*)\"");

    while (i < data_obj.size()) {
        while (i < data_obj.size() && (std::isspace(static_cast<unsigned char>(data_obj[i])) != 0 || data_obj[i] == ',')) {
            ++i;
        }
        if (i >= data_obj.size() || data_obj[i] == '}') {
            break;
        }
        if (data_obj[i] != '"') {
            break;
        }

        ++i;
        size_t key_start = i;
        while (i < data_obj.size() && data_obj[i] != '"') {
            if (data_obj[i] == '\\' && i + 1 < data_obj.size()) {
                i += 2;
            } else {
                ++i;
            }
        }
        if (i >= data_obj.size()) {
            break;
        }

        std::string key = data_obj.substr(key_start, i - key_start);
        ++i;

        while (i < data_obj.size() && (std::isspace(static_cast<unsigned char>(data_obj[i])) != 0 || data_obj[i] == ':')) {
            ++i;
        }
        if (i >= data_obj.size() || data_obj[i] != '{') {
            break;
        }

        size_t obj_start = i;
        int depth = 0;
        bool in_string = false;
        bool escaped = false;
        for (; i < data_obj.size(); ++i) {
            char c = data_obj[i];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    in_string = false;
                }
                continue;
            }
            if (c == '"') {
                in_string = true;
                continue;
            }
            if (c == '{') {
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth == 0) {
                    ++i;
                    break;
                }
            }
        }

        std::string obj = data_obj.substr(obj_start, i - obj_start);

        SolveResult r;
        if (auto v = regex_group_1(obj, re_status)) {
            r.status = *v;
        }
        if (auto v = regex_group_1(obj, re_solve_time)) {
            r.solve_time_ms = std::stod(*v);
        }
        if (auto v = regex_group_1(obj, re_steps)) {
            r.steps = std::stoi(*v);
        }
        if (auto v = regex_group_1(obj, re_solution)) {
            r.solution = *v;
        }
        if (auto v = regex_group_1(obj, re_fail_reason)) {
            r.fail_reason = *v;
        }

        if (!r.status.empty()) {
            state.results[key] = r;
        }
    }

    return state;
}

std::string get_cpu_identifier() {
#ifdef _WIN32
    return maybe_env("PROCESSOR_IDENTIFIER", "Unknown CPU");
#else
    return "Unknown CPU";
#endif
}

std::string quoted(const fs::path& p) {
    return "\"" + p.string() + "\"";
}

struct ProcessResult {
    int exit_code = -1;
    bool timed_out = false;
    double duration_ms = 0.0;
};

ProcessResult run_solver_process(const fs::path& solver_path,
                                 const fs::path& level_file,
                                 int level_idx,
                                 int timeout_sec,
                                 int solver_cores,
                                 const fs::path& out_file) {
    const auto t0 = std::chrono::steady_clock::now();

    std::ostringstream cmd;
    cmd << quoted(solver_path) << ' ' << quoted(level_file)
        << " -level " << level_idx
        << " -time " << timeout_sec
        << " -cores " << solver_cores
        << " -out_file " << quoted(out_file);

#ifdef _WIN32
    STARTUPINFOA si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    ZeroMemory(&pi, sizeof(pi));
    si.cb = sizeof(si);

    std::string cmdline = cmd.str();
    std::vector<char> mutable_cmd(cmdline.begin(), cmdline.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr,
        mutable_cmd.data(),
        nullptr,
        nullptr,
        FALSE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &si,
        &pi);

    ProcessResult pr;
    if (!ok) {
        pr.exit_code = -1;
        pr.timed_out = false;
        pr.duration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return pr;
    }

    const DWORD timeout_ms = static_cast<DWORD>((timeout_sec + 5) * 1000);
    DWORD wait_res = WaitForSingleObject(pi.hProcess, timeout_ms);

    if (wait_res == WAIT_TIMEOUT) {
        TerminateProcess(pi.hProcess, 124);
        pr.timed_out = true;
    }

    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    pr.exit_code = static_cast<int>(code);

    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    pr.duration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return pr;
#else
    int code = std::system(cmd.str().c_str());
    ProcessResult pr;
    pr.exit_code = code;
    pr.timed_out = false;
    pr.duration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return pr;
#endif
}

std::string parse_solution_line(const fs::path& out_file) {
    std::ifstream in(out_file, std::ios::binary);
    if (!in) {
        return "";
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        lines.push_back(line);
    }

    for (size_t i = 0; i < lines.size(); ++i) {
        if (trim_copy(lines[i]) == "Solution") {
            for (size_t j = i + 1; j < lines.size(); ++j) {
                std::string cand = trim_copy(lines[j]);
                if (!cand.empty()) {
                    return cand;
                }
            }
            break;
        }
    }
    return "";
}

void persist_json(const fs::path& output_path,
                  const std::vector<Task>& tasks,
                  const std::unordered_map<std::string, SolveResult>& results,
                  const PerformanceBase& base,
                  const RuntimeCounters& counters,
                  const std::chrono::steady_clock::time_point started_at) {
    const uint64_t current_processed = base.processed + counters.newly_processed.load();
    const uint64_t current_success = base.success + counters.newly_success.load();
    const uint64_t current_failed = base.failed + counters.newly_failed.load();

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started_at).count();
    const double current_wall = base.wall_clock_sec + elapsed;

    fs::path tmp_path = output_path;
    tmp_path += ".tmp";

    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "[WARN] Cannot write temp JSON: " << tmp_path << "\n";
        return;
    }

    const unsigned logical_cores = std::max(1u, std::thread::hardware_concurrency());

    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"hardware\": {\n";
    out << "      \"OS\": \"Windows 11\",\n";
    out << "      \"CPU\": \"" << json_escape(get_cpu_identifier()) << "\",\n";
    out << "      \"Logical_Cores\": " << logical_cores << ",\n";
    out << "      \"GPU\": \"Not Detected\"\n";
    out << "    },\n";
    out << "    \"performance\": {\n";
    out << "      \"total_maps_processed_this_session\": " << current_processed << ",\n";
    out << "      \"success_count_this_session\": " << current_success << ",\n";
    out << "      \"failed_count_this_session\": " << current_failed << ",\n";
    out << "      \"session_wall_clock_time_seconds\": " << std::fixed << std::setprecision(4) << current_wall << ",\n";
    out << "      \"total_maps_in_database\": " << tasks.size() << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"data\": {\n";

    bool first = true;
    for (const Task& task : tasks) {
        auto it = results.find(task.key);
        if (it == results.end()) {
            continue;
        }

        const SolveResult& r = it->second;
        if (!first) {
            out << ",\n";
        }
        first = false;

        out << "    \"" << json_escape(task.key) << "\": {\n";
        out << "      \"status\": \"" << json_escape(r.status) << "\",\n";
        out << "      \"solve_time_ms\": " << std::fixed << std::setprecision(2) << r.solve_time_ms;

        if (r.status == "success") {
            out << ",\n";
            out << "      \"steps\": " << r.steps << ",\n";
            out << "      \"solution\": \"" << json_escape(r.solution) << "\"\n";
        } else {
            if (!r.fail_reason.empty()) {
                out << ",\n";
                out << "      \"fail_reason\": \"" << json_escape(r.fail_reason) << "\"\n";
            } else {
                out << "\n";
            }
        }
        out << "    }";
    }

    out << "\n";
    out << "  }\n";
    out << "}\n";
    out.close();

    std::error_code ec;
    fs::remove(output_path, ec);
    ec.clear();
    fs::rename(tmp_path, output_path, ec);
    if (ec) {
        std::cerr << "[WARN] Cannot replace JSON output: " << ec.message() << "\n";
    }
}

struct Config {
    fs::path solver_path;
    fs::path boxoban_root = "boxoban-levels";
    fs::path output_json = "festival_boxoban_results.json";
    int timeout_sec = 600;
    int solver_cores = 0;   // 0 => auto (1|2|4|8)
    int workers = 0;        // 0 => auto
    int flush_every = 25;
};

bool parse_int_arg(const std::string& value, int* out) {
    try {
        *out = std::stoi(value);
        return true;
    } catch (...) {
        return false;
    }
}

void print_usage(const char* exe_name) {
    std::cerr << "Usage:\n";
    std::cerr << "  " << exe_name << " --solver <path_to_festival_exe> [options]\n\n";
    std::cerr << "Options:\n";
    std::cerr << "  --boxoban-root <path>   Default: boxoban-levels\n";
    std::cerr << "  --output <path>         Default: festival_boxoban_results.json\n";
    std::cerr << "  --timeout-sec <int>     Default: 600\n";
    std::cerr << "  --solver-cores <int>    1|2|4|8, default: auto\n";
    std::cerr << "  --workers <int>         Parallel workers, default: auto\n";
    std::cerr << "  --flush-every <int>     Save interval in new maps, default: 25\n";
}

std::optional<Config> parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];

        auto require_value = [&](const std::string& key) -> std::optional<std::string> {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << key << "\n";
                return std::nullopt;
            }
            ++i;
            return std::string(argv[i]);
        };

        if (a == "--solver") {
            auto v = require_value(a);
            if (!v) return std::nullopt;
            cfg.solver_path = *v;
        } else if (a == "--boxoban-root") {
            auto v = require_value(a);
            if (!v) return std::nullopt;
            cfg.boxoban_root = *v;
        } else if (a == "--output") {
            auto v = require_value(a);
            if (!v) return std::nullopt;
            cfg.output_json = *v;
        } else if (a == "--timeout-sec") {
            auto v = require_value(a);
            if (!v) return std::nullopt;
            if (!parse_int_arg(*v, &cfg.timeout_sec)) return std::nullopt;
        } else if (a == "--solver-cores") {
            auto v = require_value(a);
            if (!v) return std::nullopt;
            if (!parse_int_arg(*v, &cfg.solver_cores)) return std::nullopt;
        } else if (a == "--workers") {
            auto v = require_value(a);
            if (!v) return std::nullopt;
            if (!parse_int_arg(*v, &cfg.workers)) return std::nullopt;
        } else if (a == "--flush-every") {
            auto v = require_value(a);
            if (!v) return std::nullopt;
            if (!parse_int_arg(*v, &cfg.flush_every)) return std::nullopt;
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return std::nullopt;
        } else {
            std::cerr << "Unknown argument: " << a << "\n";
            return std::nullopt;
        }
    }

    if (cfg.solver_path.empty()) {
        std::cerr << "--solver is required.\n";
        return std::nullopt;
    }

    cfg.solver_path = fs::absolute(cfg.solver_path);
    cfg.boxoban_root = fs::absolute(cfg.boxoban_root);
    cfg.output_json = fs::absolute(cfg.output_json);

    return cfg;
}

int auto_pick_solver_cores() {
    const unsigned logical = std::max(1u, std::thread::hardware_concurrency());
    if (logical >= 32) return 8;
    if (logical >= 16) return 4;
    if (logical >= 8) return 2;
    return 1;
}

int auto_pick_workers(int solver_cores) {
    const int logical = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    int workers = logical / std::max(1, solver_cores);
    return std::max(1, workers);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#endif
    std::signal(SIGINT, signal_handler);

    auto cfg_opt = parse_args(argc, argv);
    if (!cfg_opt) {
        print_usage(argv[0]);
        return 1;
    }
    Config cfg = *cfg_opt;

    if (!fs::exists(cfg.solver_path)) {
        std::cerr << "Solver binary not found: " << cfg.solver_path << "\n";
        return 1;
    }
    if (!fs::exists(cfg.boxoban_root)) {
        std::cerr << "Boxoban root not found: " << cfg.boxoban_root << "\n";
        return 1;
    }

    if (cfg.solver_cores == 0) {
        cfg.solver_cores = auto_pick_solver_cores();
    }
    if (!(cfg.solver_cores == 1 || cfg.solver_cores == 2 || cfg.solver_cores == 4 || cfg.solver_cores == 8)) {
        std::cerr << "--solver-cores must be one of 1, 2, 4, 8\n";
        return 1;
    }

    if (cfg.workers == 0) {
        cfg.workers = auto_pick_workers(cfg.solver_cores);
    }
    cfg.workers = std::max(1, cfg.workers);
    cfg.flush_every = std::max(1, cfg.flush_every);

    std::cout << "[INFO] Scanning Boxoban files under: " << cfg.boxoban_root << "\n";
    std::vector<fs::path> files = collect_numeric_level_files(cfg.boxoban_root);
    std::cout << "[INFO] Numeric .txt files found: " << files.size() << "\n";

    std::cout << "[INFO] Building map task list...\n";
    std::vector<Task> tasks = build_tasks(files);
    std::cout << "[INFO] Total maps in database: " << tasks.size() << "\n";

    LoadState loaded = load_existing_json(cfg.output_json);

    std::unordered_map<std::string, SolveResult> results;
    results.reserve(tasks.size());

    for (const auto& kv : loaded.results) {
        results[kv.first] = kv.second;
    }

    std::vector<char> completed(tasks.size(), 0);
    size_t already_done = 0;
    for (size_t i = 0; i < tasks.size(); ++i) {
        if (results.find(tasks[i].key) != results.end()) {
            completed[i] = 1;
            ++already_done;
        }
    }

    std::cout << "[INFO] Existing output detected: " << (fs::exists(cfg.output_json) ? "yes" : "no") << "\n";
    std::cout << "[INFO] Resume progress: " << already_done << " / " << tasks.size() << " already processed\n";
    std::cout << "[INFO] Solver cores per process: " << cfg.solver_cores << "\n";
    std::cout << "[INFO] Worker processes: " << cfg.workers << "\n";

    fs::path temp_dir = cfg.output_json.parent_path() / "festival_batch_tmp";
    std::error_code ec;
    fs::create_directories(temp_dir, ec);

    std::mutex results_mutex;
    std::atomic<size_t> task_cursor{0};
    RuntimeCounters counters;

    const auto started_at = std::chrono::steady_clock::now();

    auto flush_snapshot = [&]() {
        std::unordered_map<std::string, SolveResult> snapshot;
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            snapshot = results;
        }
        persist_json(cfg.output_json, tasks, snapshot, loaded.perf_base, counters, started_at);
    };

    auto worker_fn = [&](int worker_id) {
        fs::path out_file = temp_dir / ("worker_" + std::to_string(worker_id) + ".sok");

        while (!g_stop_requested.load()) {
            size_t idx = task_cursor.fetch_add(1);
            if (idx >= tasks.size()) {
                break;
            }

            if (completed[idx]) {
                continue;
            }

            const Task& task = tasks[idx];

            ProcessResult pr = run_solver_process(cfg.solver_path,
                                                  task.level_file,
                                                  task.level_index_1based,
                                                  cfg.timeout_sec,
                                                  cfg.solver_cores,
                                                  out_file);

            SolveResult sr;
            sr.solve_time_ms = pr.duration_ms;

            std::string solution = parse_solution_line(out_file);
            if (!solution.empty()) {
                sr.status = "success";
                sr.solution = solution;
                sr.steps = static_cast<int>(solution.size());
            } else {
                sr.status = "failed";
                sr.steps = 0;

                if (pr.timed_out) {
                    sr.fail_reason = "timeout";
                } else if (pr.exit_code != 0) {
                    sr.fail_reason = "solver_exit_" + std::to_string(pr.exit_code);
                } else {
                    sr.fail_reason = "no_solution";
                }
            }

            bool should_flush = false;
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                results[task.key] = sr;

                uint64_t p = counters.newly_processed.fetch_add(1) + 1;
                if (sr.status == "success") {
                    counters.newly_success.fetch_add(1);
                } else {
                    counters.newly_failed.fetch_add(1);
                }

                if ((p % static_cast<uint64_t>(cfg.flush_every)) == 0) {
                    should_flush = true;
                }
            }

            if (should_flush) {
                flush_snapshot();
            }

            const uint64_t done = counters.newly_processed.load();
            if ((done % 10) == 0) {
                std::cout << "[INFO] Session processed: " << done
                          << " | success=" << counters.newly_success.load()
                          << " failed=" << counters.newly_failed.load() << "\n";
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(cfg.workers));
    for (int i = 0; i < cfg.workers; ++i) {
        workers.emplace_back(worker_fn, i);
    }

    for (auto& t : workers) {
        t.join();
    }

    flush_snapshot();

    std::cout << "[INFO] Final session stats: processed=" << counters.newly_processed.load()
              << " success=" << counters.newly_success.load()
              << " failed=" << counters.newly_failed.load() << "\n";

    if (g_stop_requested.load()) {
        std::cout << "[INFO] Interrupted by Ctrl+C. Progress was saved to: " << cfg.output_json << "\n";
        return 130;
    }

    std::cout << "[INFO] Completed. JSON output: " << cfg.output_json << "\n";
    return 0;
}
