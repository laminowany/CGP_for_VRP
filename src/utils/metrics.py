import pandas as pd

class MetricsLogger:
    def __init__(self):
        self.records = []

    def record(self, **kwargs):
        self.records.append(kwargs)

    def to_dataframe(self):

        return pd.DataFrame(self.records)

    def save_csv(self, path):
        df = self.to_dataframe()
        try:
            existing_df = pd.read_csv(path)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass 
        df.to_csv(path, index=False)