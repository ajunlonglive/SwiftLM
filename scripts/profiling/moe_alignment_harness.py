import sys
import json
import time
import urllib.request
import urllib.error

# Dedicated harness for verifying the 4x MoE performance speedup on gemma-4-26b-a4b-it-4bit
# Bypasses existing benchmark overhead; deeply profiles multi-token generation specifically.

MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
PORT = 5431
ENDPOINT = f"http://127.0.0.1:{PORT}/v1/chat/completions"

print(f"==================================================")
print(f"  Gemma 4 MoE Performance Alignment Harness")
print(f"==================================================")
print(f"Targeting port {PORT} with model: {MODEL}")

# Health check
try:
    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
    print("Server is up and responding to health checks.\n")
except Exception as e:
    print(f"❌ Cannot reach SwiftLM server on port {PORT}. Ensure it is running.")
    sys.exit(1)

# Large prompt generation
print("Preparing generation payload...\n")
payload = {
    "model": MODEL,
    "max_tokens": 200,
    "temperature": 0.0,
    "messages": [
        {"role": "user", "content": "Write a highly detailed, very long story about a cybernetic owl named Ozymandias who lives in a neo-noir futuristic city. Include dialogue and descriptions of the technology surrounding him."}
    ],
    "stream": True # Important: we want to stream and measure TPS live
}

req = urllib.request.Request(
    ENDPOINT,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

print("Starting generation and capturing exact TPS...")
start_time = time.time()
ttft = None
tokens_generated = 0

try:
    with urllib.request.urlopen(req) as response:
        for line in response:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
                
            data_str = line[len("data: "):]
            if data_str == "[DONE]":
                break
                
            # Got a token
            if ttft is None:
                ttft = time.time() - start_time
                print(f"⏱️ TTFT: {ttft:.3f} seconds\n")
                print("Generating: ", end="", flush=True)
                
            chunk = json.loads(data_str)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                print(content, end="", flush=True)
                tokens_generated += 1

except urllib.error.URLError as e:
    print(f"❌ Error during generation: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    sys.exit(1)

end_time = time.time()
generation_time = end_time - (start_time + (ttft if ttft else 0))
tps = tokens_generated / generation_time if generation_time > 0 else 0

print("\n\n==================================================")
print("  RESULTS")
print("==================================================")
print(f"Total tokens generated: {tokens_generated}")
print(f"Generation time:        {generation_time:.2f} seconds")
print(f"Tokens Per Second:      {tps:.2f} tok/s")
print("==================================================")

if tps >= 80:
    print("✅ SUCCESS: Target performance reached (Aligned with Osaurus fork).")
else:
    print("⚠️ WARNING: Target performance may not have been reached.")
