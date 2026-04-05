import argparse
import subprocess
import time
import urllib.request
import urllib.error
import json
import re
import sys
import os

CONFIGS = [
    {"name": "Dense/Vanilla", "flags": []},
    {"name": "SSD Stream", "flags": ["--stream-experts"]},
    {"name": "TurboQuant", "flags": ["--turbo-kv"]},
    {"name": "SSD + TurboQuant", "flags": ["--stream-experts", "--turbo-kv"]}
]

SWIFTLM_PATH = ".build/arm64-apple-macosx/release/SwiftLM"

def poll_health(port=5413, timeout=30):
    start = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - start < timeout:
        try:
            r = urllib.request.urlopen(url)
            if r.getcode() == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def make_request_stream(prompt_len, max_tokens, port=5413):
    prompt = "apple " * int(prompt_len * 0.75)
    data = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    ttft = None
    start = time.time()
    tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    if ttft is None:
                        ttft = time.time() - start
                    tokens += 1
            total_time = time.time() - start
            gen_time = total_time - ttft if ttft else 0
            tps = (tokens - 1) / gen_time if gen_time > 0 and tokens > 1 else 0
            return True, ttft, tps
    except Exception as e:
        print(f"Request failed: {e}")
        return False, 0, 0

def extract_base_memory(log_path):
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if "Memory strategy: FULL GPU" in line:
                    m = re.search(r"\(([0-9.]+)GB model", line)
                    if m: return f"{m.group(1)} GB"
    except: pass
    return "N/A"

def extract_real_memory(log_path):
    try:
        with open(log_path, 'r') as f:
            log_data = f.read()
            m = re.findall(r"OS_RAM=([0-9.]+)", log_data)
            if m: return f"{m[-1]} GB"
    except: pass
    return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Aegis-AI Physical Model Profiler")
    parser.add_argument("--model", required=True, help="Model ID (e.g. gemma-4-26b-a4b-it-4bit)")
    parser.add_argument("--out", default="./profiling_results.md", help="Output markdown file path")
    args = parser.parse_args()
    
    results = []
    subprocess.run(["killall", "SwiftLM"], stderr=subprocess.DEVNULL)
    
    for config in CONFIGS:
        print(f"\n--- Profiling {args.model} [{config['name']}] ---")
        model_path = f"/Users/simba/.aegis-ai/models/mlx_models/mlx-community/{args.model}"
        
        log_path = "./tmp/profile_server.log"
        cmd = [SWIFTLM_PATH, "--model", model_path] + config["flags"]
        
        with open(log_path, "w") as root_log:
            server_proc = subprocess.Popen(cmd, stdout=root_log, stderr=subprocess.STDOUT)
        
        if not poll_health():
            print("Server failed to start.")
            server_proc.terminate()
            continue
            
        static_mem = extract_base_memory(log_path)
        
        print("Running 20-token test (prefill ~512, max ~20)...")
        ok, ttft, tps = make_request_stream(prompt_len=512, max_tokens=20)
        
        server_proc.send_signal(subprocess.signal.SIGTERM)
        server_proc.wait(timeout=10)
        
        real_mem = extract_real_memory(log_path)
        
        if ok:
            results.append({
                "config": config["name"],
                "ttft_20": f"{ttft:.2f}",
                "tps_20": f"{tps:.2f}",
                "static_mem": static_mem,
                "real_mem": real_mem
            })
            print(f"Result [{config['name']}]: TTFT={ttft:.2f}s TPS={tps:.2f} BaseRAM={static_mem} PhysRAM={real_mem}")
        
    with open(args.out, "w") as f:
        f.write(f"### `{args.model}` - Throughput & OS Memory Profile\n\n")
        f.write("| Configuration | Time To First Token | Generation Speed | Theoretical Reservation | Physical OS Footprint (RAM) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['config']} | {r['ttft_20']}s | {r['tps_20']} tok/s | {r['static_mem']} | {r['real_mem']} |\n")
            
    print(f"\nDone. Results saved to {args.out}")

if __name__ == "__main__":
    main()
