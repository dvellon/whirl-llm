"""FNMX: chat (zh/en code), vision (2 images) and MTP-acceptance checks against a running llama-server.
Usage: python gpu_tests.py PORT OUT.json"""
import json, sys, base64, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
E = r"C:\ClaudeWorkspace\models\fnmx\eval"
def post(msgs, max_tokens=400):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False}, "timings_per_token": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    return {"text": r["choices"][0]["message"]["content"], "usage": r.get("usage"), "timings": r.get("timings"), "wall_s": round(time.time() - t, 2)}
def img(p):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}}
tests = {
  "zh_code": [{"role": "user", "content": "請用 Python 寫一個函式 merge_intervals(intervals)，合併重疊的區間並回傳排序後的結果。程式碼用英文命名，並用繁體中文簡短說明演算法與時間複雜度。"}],
  "en_code": [{"role": "user", "content": "Write a C++17 function that returns the k most frequent words in a std::vector<std::string>, ties broken alphabetically. Keep it short and explain the complexity in one sentence."}],
  "vision_en": [{"role": "user", "content": [img(E + r"\img_shapes_en.png"), {"type": "text", "text": "Describe this image. Quote any text exactly."}]}],
  "vision_zh": [{"role": "user", "content": [img(E + r"\img_sign_zh.png"), {"type": "text", "text": "請逐字讀出圖片中的所有文字，然後用一句話說明這是什麼。"}]}],
}
res = {}
for k, m in tests.items():
    res[k] = post(m, 500 if "code" in k else 300)
    print(k, res[k]["wall_s"], res[k]["timings"], flush=True)
json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
