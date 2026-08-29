def validate_numeric_result(value, *, minimum=None, maximum=None):
    checks = []
    if minimum is not None:
        checks.append(value >= minimum)
    if maximum is not None:
        checks.append(value <= maximum)
    return {
        "valid": all(checks) if checks else True,
        "checks": checks,
    }
