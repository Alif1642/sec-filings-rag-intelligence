from __future__ import annotations

import argparse
import json

from evals.run_eval import EvaluationRunner

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='evals/datasets/sample_questions.json')
args = parser.parse_args()
print(json.dumps(EvaluationRunner().run(args.dataset), indent=2))
