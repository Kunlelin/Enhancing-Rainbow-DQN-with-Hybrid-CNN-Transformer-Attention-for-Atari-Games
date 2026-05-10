import os
import json
import csv
import time
from collections import defaultdict


class Logger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.csv_file = os.path.join(self.log_dir, "metrics.csv")
        self.csv_header_written = False

    def log(self, step, **kwargs):
        row = {"step": step, "time": time.time() - self.start_time}
        row.update(kwargs)
        for k, v in kwargs.items():
            self.metrics[k].append((step, v))
        if not self.csv_header_written:
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            self.csv_header_written = True
        else:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

    def save_config(self, config):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def get_metrics(self):
        return dict(self.metrics)

    def save_metrics(self):
        with open(os.path.join(self.log_dir, "metrics.json"), "w") as f:
            json.dump(dict(self.metrics), f)
