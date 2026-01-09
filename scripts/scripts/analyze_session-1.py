import json
import subprocess
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

PROFILE = Path("profiles/profile.json")
PROFILE.parent.mkdir(exist_ok=True)

FIELDS = [
    ("name", "姓名"),
    ("height", "身高(cm)"),
    ("weight", "體重(kg)"),
    ("upper_1rm", "上肢1RM"),
    ("lower_1rm", "下肢1RM"),
    ("bmi", "BMI"),
    ("pbf", "體脂率%"),
    ("smm", "骨骼肌量(kg)")
]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rowing HRV System")

        self.vars = {}
        for k, label in FIELDS:
            tk.Label(self, text=label).pack()
            v = tk.StringVar()
            tk.Entry(self, textvariable=v).pack()
            self.vars[k] = v

        tk.Button(self, text="保存 Profile", command=self.save).pack(pady=4)
        tk.Button(self, text="開始量測", command=self.run).pack(pady=4)
        tk.Button(self, text="HRV 分析", command=self.analyze).pack(pady=4)

    def save(self):
        data = {k: v.get() for k, v in self.vars.items()}
        PROFILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        messagebox.showinfo("OK", "Profile 已保存")

    def run(self):
        subprocess.Popen(["python", "scripts/run_local.py"])
        messagebox.showinfo("Info", "量測中，結束請按 q")

    def analyze(self):
        subprocess.call(["python", "scripts/analyze_session_hrv.py"])
        messagebox.showinfo("OK", "HRV 分析完成")

if __name__ == "__main__":
    App().mainloop()
