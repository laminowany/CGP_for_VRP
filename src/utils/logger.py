import os
import json
import pandas as pd

EXCLUDE = {"device"}

class Logger:
    def __init__(self, opts):
        self.records = {} 
        self.opts = opts
        self.saved_args = False
     
    def record(self, key="epochs", **kwargs):
        if not self.saved_args:
            with open(os.path.join(self.opts.save_dir, "args.json"), 'w') as f:
                json.dump({k: v for k, v in vars(self.opts).items() if k not in EXCLUDE}, f, indent=2)
            self.saved_args = True
        if key not in self.records:
            self.records[key] = []
        self.records[key].append(kwargs)
        path = os.path.join(self.opts.save_dir, f"{key}.csv")
        os.makedirs(self.opts.save_dir, exist_ok=True)
        df = pd.DataFrame([kwargs])
        df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)

    def to_dataframe(self, key):
        return pd.DataFrame(self.records[key])
