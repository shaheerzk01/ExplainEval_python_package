def flatten_report(report):
    return {f"{k}_{subk}": v for k, v in report.items() if isinstance(v, dict) for subk, v in v.items()}
