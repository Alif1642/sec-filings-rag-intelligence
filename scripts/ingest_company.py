from __future__ import annotations

import argparse
import json

from ingestion.pipeline import IngestionPipeline

parser = argparse.ArgumentParser()
parser.add_argument('--ticker', required=True)
parser.add_argument('--form', default='10-K', choices=['10-K', '10-Q'])
parser.add_argument('--filing-date', default=None)
args = parser.parse_args()
print(json.dumps(IngestionPipeline().ingest(args.ticker, args.form, args.filing_date), indent=2))
