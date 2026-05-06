import os, re, sys, threading, subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = sys.executable
CONTROL_FILE = PROJECT_ROOT / "run_control.txt"

def discover_local_sc2_maps() -> list[str]:
    roots = [
        Path.home() / "Documents" / "StarCraft II" / "Maps",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "StarCraft II" / "Maps",
        Path(os.environ.get("PROGRAMFILES", "")) / "StarCraft II" / "Maps",
    ]
    names = set()
    for r in roots:
        if r.exists():
            for p in r.rglob("*.SC2Map"):
                names.add(p.stem)
    return sorted(names)

class SimulationUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("无人集群协同对抗仿真系统")
        self.root.geometry("1240x840")
        self.process: subprocess.Popen | None = None
        self.maps = discover_local_sc2_maps()
        self.metric_vars = {k: tk.StringVar(value="-") for k in [
            "mode", "device", "episode", "avg_reward", "avg_win", "eval_reward", "eval_win", "eval_len", "battle", "ckpt"
        ]}
        self._build_layout()

    def _build_layout(self):
        main = ttk.Frame(self.root); main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(main); left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(main); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self._build_controls(left); self._build_metrics(right); self._build_log(right)

    def _build_controls(self, p):
        top = ttk.LabelFrame(p, text="仿真参数"); top.pack(fill=tk.X)
        self.map_var = tk.StringVar(value=self.maps[0] if self.maps else "")
        self.algo_var = tk.StringVar(value="happo")
        self.eval_var = tk.BooleanVar(value=False)
        self.visualize_var = tk.BooleanVar(value=False)
        self.model_var = tk.StringVar(value="")
        self.save_dir_var = tk.StringVar(value=str(PROJECT_ROOT / "checkpoints"))
        self.run_name_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="auto")
        self.num_agents_var = tk.StringVar(value="")
        self.episodes_var = tk.StringVar(value="200")
        self.eval_episodes_var = tk.StringVar(value="1")
        self.max_steps_var = tk.StringVar(value="3000")
        self.step_mul_train_var = tk.StringVar(value="16")
        self.step_mul_eval_var = tk.StringVar(value="2")
        self.lr_var = tk.StringVar(value="0.0005")
        self.clip_var = tk.StringVar(value="0.2")
        self.update_var = tk.StringVar(value="1000")
        self.save_interval_var = tk.StringVar(value="20")
        self.num_collect_envs_var = tk.StringVar(value="1")
        self.hold_seconds_var = tk.StringVar(value="8")
        self.threads_var = tk.StringVar(value="0")
        self.debug_force_spread_var = tk.BooleanVar(value=False)

        self._sync_visualize_state()

        r = 0
        ttk.Label(top, text="地图").grid(row=r, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(top, textvariable=self.map_var, values=self.maps, width=24).grid(row=r, column=1, sticky="w", padx=4)
        ttk.Label(top, text="算法").grid(row=r, column=2, sticky="e", padx=4)
        ttk.Combobox(top, textvariable=self.algo_var, values=["happo", "mappo", "simplebattle"], width=12, state="readonly").grid(row=r, column=3, sticky="w", padx=4)
        r += 1
        ttk.Checkbutton(top, text="评估模式", variable=self.eval_var, command=self._sync_visualize_state).grid(row=r, column=0, columnspan=2, sticky="w", padx=4)
        self.visualize_check = ttk.Checkbutton(top, text="显示画面", variable=self.visualize_var)
        self.visualize_check.grid(row=r, column=2, sticky="w", padx=4)
        ttk.Checkbutton(top, text="调试:前30步强制分散", variable=self.debug_force_spread_var).grid(row=r, column=3, sticky="w", padx=4)

        rows = [
            ("训练回合", self.episodes_var, "评估局数", self.eval_episodes_var),
            ("最大步数", self.max_steps_var, "智能体数", self.num_agents_var),
            ("训练速度 step_mul", self.step_mul_train_var, "评估速度 step_mul", self.step_mul_eval_var),
            ("lr", self.lr_var, "clip", self.clip_var),
            ("update_interval", self.update_var, "save_interval", self.save_interval_var),
            ("并行采样环境", self.num_collect_envs_var, "device", self.device_var),
            ("结束保留秒数", self.hold_seconds_var, "torch_threads", self.threads_var),
        ]
        for l1, v1, l2, v2 in rows:
            r += 1
            ttk.Label(top, text=l1).grid(row=r, column=0, sticky="e", padx=4)
            ttk.Entry(top, textvariable=v1, width=12).grid(row=r, column=1, sticky="w")
            ttk.Label(top, text=l2).grid(row=r, column=2, sticky="e", padx=4)
            if l2 == "device":
                ttk.Combobox(top, textvariable=v2, values=["auto", "cuda", "cpu"], width=10, state="readonly").grid(row=r, column=3, sticky="w")
            else:
                ttk.Entry(top, textvariable=v2, width=12).grid(row=r, column=3, sticky="w")

        r += 1
        ttk.Label(top, text="模型(文件/目录)").grid(row=r, column=0, sticky="e", padx=4)
        ttk.Entry(top, textvariable=self.model_var, width=30).grid(row=r, column=1, columnspan=2, sticky="w")
        ttk.Button(top, text="浏览", command=self._browse_model).grid(row=r, column=3, sticky="w")
        r += 1
        ttk.Label(top, text="保存目录").grid(row=r, column=0, sticky="e", padx=4)
        ttk.Entry(top, textvariable=self.save_dir_var, width=30).grid(row=r, column=1, columnspan=2, sticky="w")
        ttk.Button(top, text="选择", command=self._browse_save_dir).grid(row=r, column=3, sticky="w")
        r += 1
        ttk.Label(top, text="运行名").grid(row=r, column=0, sticky="e", padx=4)
        ttk.Entry(top, textvariable=self.run_name_var, width=20).grid(row=r, column=1, sticky="w")

        b = ttk.Frame(p); b.pack(fill=tk.X, pady=(8, 0))
        ttk.Button(b, text="启动", command=self.start_simulation).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="保存并停止", command=self.save_and_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="强制停止", command=self.stop_simulation).pack(side=tk.LEFT, padx=4)
        ttk.Button(b, text="清空日志", command=self.clear_log).pack(side=tk.LEFT, padx=4)

    def _build_metrics(self, p):
        box = ttk.LabelFrame(p, text="训练/评估数据"); box.pack(fill=tk.X)
        items = [
            ("模式", "mode"), ("设备", "device"), ("轮次", "episode"),
            ("训练平均奖励", "avg_reward"), ("训练平均胜率", "avg_win"),
            ("评估奖励", "eval_reward"), ("评估胜率", "eval_win"), ("评估步长", "eval_len"),
            ("对战结果", "battle"), ("最近checkpoint", "ckpt"),
        ]
        for i, (n, k) in enumerate(items):
            ttk.Label(box, text=n).grid(row=i, column=0, sticky="e", padx=4, pady=2)
            ttk.Label(box, textvariable=self.metric_vars[k], width=60).grid(row=i, column=1, sticky="w", padx=4, pady=2)

    def _build_log(self, p):
        log = ttk.LabelFrame(p, text="控制台输出"); log.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.log_text = tk.Text(log, wrap=tk.WORD, bg="#0f111a", fg="#d4d4d4", insertbackground="white")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log, orient=tk.VERTICAL, command=self.log_text.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=sb.set)

    def _sync_visualize_state(self):
        if self.eval_var.get():
            self.visualize_var.set(True)
            if hasattr(self, "visualize_check"):
                self.visualize_check.state(["disabled"])
        else:
            self.visualize_var.set(False)
            if hasattr(self, "visualize_check"):
                self.visualize_check.state(["!disabled"])

    def _browse_model(self):
        # 支持两种方式：
        # 1) 选择单个 .pth（兼容旧逻辑）
        # 2) 选择目录（用于每个智能体一个 actor 权重文件）
        mode = messagebox.askyesno("选择模型来源", "是否选择“模型目录”？\n是：目录（每个单位一个actor）\n否：单个.pth文件")
        if mode:
            p = filedialog.askdirectory(title="选择模型目录", initialdir=str(PROJECT_ROOT))
        else:
            p = filedialog.askopenfilename(title="选择模型文件", filetypes=[("PyTorch", "*.pth"), ("All", "*.*")], initialdir=str(PROJECT_ROOT))
        if p:
            self.model_var.set(p)

    def _browse_save_dir(self):
        p = filedialog.askdirectory(title="选择保存目录", initialdir=str(PROJECT_ROOT))
        if p: self.save_dir_var.set(p)

    def _append_log(self, t: str):
        self.log_text.insert(tk.END, t); self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def _build_command(self) -> list[str]:
        is_eval = self.eval_var.get()
        step_mul = self.step_mul_eval_var.get().strip() if is_eval else self.step_mul_train_var.get().strip()
        cmd = [PYTHON_EXE, str(PROJECT_ROOT / "main.py")]
        cmd += ["--map", self.map_var.get().strip(), "--algo", self.algo_var.get().strip()]
        cmd += ["--episodes", self.episodes_var.get().strip(), "--eval-episodes", self.eval_episodes_var.get().strip()]
        cmd += ["--max-steps", self.max_steps_var.get().strip(), "--step-mul", step_mul]
        cmd += ["--lr", self.lr_var.get().strip(), "--clip-param", self.clip_var.get().strip()]
        cmd += ["--update-interval", self.update_var.get().strip(), "--save-interval", self.save_interval_var.get().strip()]
        cmd += ["--save-dir", self.save_dir_var.get().strip(), "--device", self.device_var.get().strip()]
        cmd += ["--num-collect-envs", self.num_collect_envs_var.get().strip()]
        cmd += ["--hold-seconds", self.hold_seconds_var.get().strip()]
        cmd += ["--torch-threads", self.threads_var.get().strip(), "--control-file", str(CONTROL_FILE)]
        if is_eval and self.visualize_var.get(): cmd += ["--visualize"]
        if self.debug_force_spread_var.get(): cmd += ["--debug-force-spread"]
        x = self.num_agents_var.get().strip()
        if x: cmd += ["--num-agents", x]
        x = self.run_name_var.get().strip()
        if x: cmd += ["--run-name", x]
        x = self.model_var.get().strip()
        if x: cmd += ["--model-path", x]
        if is_eval: cmd += ["--eval"]
        return cmd

    def start_simulation(self):
        if self.process and self.process.poll() is None:
            messagebox.showwarning("提示", "已有进程运行中"); return
        if not self.map_var.get().strip():
            messagebox.showwarning("提示", "未找到本地 SC2 地图"); return
        if CONTROL_FILE.exists(): CONTROL_FILE.unlink(missing_ok=True)
        cmd = self._build_command()
        self.metric_vars["mode"].set("评估" if self.eval_var.get() else "训练")
        self.metric_vars["battle"].set("-")
        self._append_log("\n=== 启动命令 ===\n" + " ".join(cmd) + "\n\n")
        try:
            self.process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        except Exception as e:
            messagebox.showerror("启动失败", str(e)); return
        threading.Thread(target=self._stream_output, daemon=True).start()

    def _parse_metrics(self, line: str):
        m = re.search(r"Using device:\s*(\w+)", line)
        if m: self.metric_vars["device"].set(m.group(1))
        m = re.search(r"TRAIN_METRIC\s+episode=(\d+)\s+avg_reward=([-0-9.]+)\s+avg_win_rate=([-0-9.]+)", line)
        if m:
            self.metric_vars["episode"].set(m.group(1)); self.metric_vars["avg_reward"].set(m.group(2)); self.metric_vars["avg_win"].set(m.group(3))
        m = re.search(r"EVAL_METRIC\s+avg_reward=([-0-9.]+)\s+avg_win_rate=([-0-9.]+)\s+avg_length=([-0-9.]+)", line)
        if m:
            self.metric_vars["eval_reward"].set(m.group(1)); self.metric_vars["eval_win"].set(m.group(2)); self.metric_vars["eval_len"].set(m.group(3))
        m = re.search(r"EVAL_BATTLE\s+winner=(\w+)\s+p1_reward=([-0-9.]+)\s+p2_reward=([-0-9.]+)\s+p1_score=([-0-9.]+)\s+p2_score=([-0-9.]+)\s+steps=(\d+)", line)
        if m:
            self.metric_vars["battle"].set(f"winner={m.group(1)}, p1={m.group(2)}, p2={m.group(3)}, score=({m.group(4)},{m.group(5)}), steps={m.group(6)}")
        m = re.search(r"(Model saved at episode .*: .*\.pth|Manual save done: .*\.pth)", line)
        if m: self.metric_vars["ckpt"].set(m.group(1))

    def _stream_output(self):
        assert self.process is not None
        for line in self.process.stdout:
            self.root.after(0, self._append_log, line)
            self.root.after(0, self._parse_metrics, line)
        code = self.process.wait()
        self.root.after(0, self._append_log, f"\n=== 进程结束，退出码: {code} ===\n")

    def save_and_stop(self):
        if not self.process or self.process.poll() is not None:
            self._append_log("\n=== 当前没有运行中的进程 ===\n"); return
        with open(CONTROL_FILE, "w", encoding="utf-8") as f: f.write("save_and_stop")
        self._append_log("\n=== 已请求保存并停止（安全退出） ===\n")

    def stop_simulation(self):
        if self.process and self.process.poll() is None:
            self.process.terminate(); self._append_log("\n=== 已强制停止进程 ===\n")
        else:
            self._append_log("\n=== 当前没有运行中的进程 ===\n")


def main():
    root = tk.Tk(); app = SimulationUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_simulation(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()
