import json
import os

FILES = ["last_signals_v60a.json", "last_signals_v70a.json"]

def load(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def cleanup():
    for fname in FILES:
        data = load(fname)
        if not data:
            print(f"⏩ {fname} خالی است.")
            continue
        
        # پیدا کردن معاملات باز (غیر __)
        open_keys = [k for k, v in data.items()
                     if not k.startswith("__") and v.get("status") == "OPEN"]
        
        if len(open_keys) <= 1:
            print(f"✅ {fname}: {len(open_keys)} معامله باز دارد، تغییری لازم نیست.")
            continue
        
        # نگه‌داشتن اولین معامله و حذف بقیه
        keep = open_keys[0]
        removed = open_keys[1:]
        
        for k in removed:
            del data[k]
            print(f"🗑️ حذف شد: {k} از {fname}")
        
        save(fname, data)
        print(f"💾 {fname} ذخیره شد (۱ معامله باز نگه داشته شد: {keep}).\n")

if __name__ == "__main__":
    print("🚀 شروع پاک‌سازی معاملات اضافی...\n")
    cleanup()
    print("🎉 تمام شد.")