#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from agent0.core.config import DEFAULT_MODEL, DEFAULT_PROVIDER, SUPPORTED_PROVIDERS, resolve_runtime
from agent0.core.logger import setup_logger
from agent0.curriculum.planner import plan_curriculum
from agent0.core.utils import write_json

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Curriculum Planner")
    parser.add_argument("--domain", type=str, default="coding")
    parser.add_argument("--total_tasks", type=int, default=30)
    parser.add_argument("--reflection_rounds", type=int, default=1)
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--output", type=str, default="plan.json")
    parser.add_argument("--feedback_from", type=str, default=None,
                        help="summary.json của run_curriculum (có --uncertainty_samples): p_hat theo subtopic gửi cho planner")
    parser.add_argument("--quiet", action="store_true")
    return parser

def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    setup_logger(quiet=args.quiet)
    
    runtime = resolve_runtime(args.provider, args.model, args.base_url, args.api_key)
    
    feedback = None
    if args.feedback_from:
        from agent0.curriculum.uncertainty import load_feedback

        feedback = load_feedback(args.feedback_from)
    plan = plan_curriculum(runtime, args.domain, args.total_tasks, args.reflection_rounds, feedback=feedback)
    write_json(Path(args.output), plan)  # write_json cần Path (bản cũ truyền str -> lỗi khi ghi)

if __name__ == "__main__":
    main()
