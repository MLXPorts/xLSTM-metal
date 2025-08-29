import csv
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class TelemetryLogger:
    """Run-wide telemetry logger (CSV and/or JSONL).

    Usage:
        log = TelemetryLogger(out_dir='runs/telem', csv_name='run.csv')
        log.log(step=0, **telem_dict)
        log.close()
    """

    def __init__(self, out_dir: str, csv_name: str = 'telemetry.csv', jsonl_name: Optional[str] = None):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out / csv_name
        self.jsonl_path = self.out / jsonl_name if jsonl_name else None
        self._csv_file = None
        self._csv_writer = None
        self._fields = None

    def log(self, step: int, **fields: Dict[str, float]):
        fields = {**fields}
        fields['step'] = int(step)
        fields['ts'] = datetime.utcnow().isoformat(timespec='seconds')
        # CSV lazy init with discovered header
        if self._csv_writer is None:
            self._fields = list(fields.keys())
            self._csv_file = self.csv_path.open('w', newline='')
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fields)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(fields)
        if self.jsonl_path is not None:
            with self.jsonl_path.open('a') as jf:
                jf.write(json.dumps(fields) + "\n")

    def close(self):
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

