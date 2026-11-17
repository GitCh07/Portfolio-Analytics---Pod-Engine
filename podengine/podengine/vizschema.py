"""Validate viz/data/*.json against viz/data/schema.json.

Uses the `jsonschema` package when installed; otherwise a small built-in
validator covering the keywords the schema uses. Also checks the cross-field
rules a JSON Schema cannot express (matching array lengths)."""
import json
import os
import re

from .vizexport import VIZ_DIR


def _check(v, s, root, path, errs):
    if "$ref" in s:
        node = root
        for part in s["$ref"].lstrip("#/").split("/"):
            node = node[part]
        return _check(v, node, root, path, errs)
    t = s.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = any((tt == "object" and isinstance(v, dict)) or (tt == "array" and isinstance(v, list)) or
                 (tt == "string" and isinstance(v, str)) or (tt == "boolean" and isinstance(v, bool)) or
                 (tt == "null" and v is None) or
                 (tt == "integer" and isinstance(v, int) and not isinstance(v, bool)) or
                 (tt == "number" and isinstance(v, (int, float)) and not isinstance(v, bool)) for tt in types)
        if not ok:
            errs.append(f"{path}: expected {t}, got {type(v).__name__}")
            return
    if "enum" in s and v not in s["enum"]:
        errs.append(f"{path}: {v!r} not in {s['enum']}")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if "minimum" in s and v < s["minimum"]:
            errs.append(f"{path}: {v} < {s['minimum']}")
        if "maximum" in s and v > s["maximum"]:
            errs.append(f"{path}: {v} > {s['maximum']}")
        if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]:
            errs.append(f"{path}: {v} <= {s['exclusiveMinimum']}")
    if isinstance(v, str) and "pattern" in s and not re.search(s["pattern"], v):
        errs.append(f"{path}: {v!r} does not match {s['pattern']}")
    if isinstance(v, dict):
        for k in s.get("required", []):
            if k not in v:
                errs.append(f"{path}: missing '{k}'")
        props = s.get("properties", {})
        for k, x in v.items():
            if k in props:
                _check(x, props[k], root, f"{path}.{k}", errs)
            elif isinstance(s.get("additionalProperties"), dict):
                _check(x, s["additionalProperties"], root, f"{path}.{k}", errs)
    if isinstance(v, list):
        if len(v) < s.get("minItems", 0):
            errs.append(f"{path}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            errs.append(f"{path}: more than {s['maxItems']} items")
        if "items" in s:
            for i, x in enumerate(v):
                _check(x, s["items"], root, f"{path}[{i}]", errs)


def consistency(name, obj):
    """Rules JSON Schema cannot express."""
    errs = []
    if name == "fan.json":
        h = obj["horizon_days"]
        for i, f in enumerate(obj["frames"]):
            if len(f["realized"]) != h + 1 or any(len(b) != h + 1 for b in f["bands"].values()):
                errs.append(f"fan frame {i}: series must have horizon_days + 1 points")
            b = [f["bands"][str(q)] for q in obj["percentiles"]]
            if any(b[j][k] > b[j + 1][k] + 1e-12 for j in range(len(b) - 1) for k in range(len(b[0]))):
                errs.append(f"fan frame {i}: percentile bands out of order")
    if name == "calibration.json":
        for i, f in enumerate(obj["frames"]):
            if sum(f["counts"]) != f["n"] or len(f["counts"]) != obj["bins"]:
                errs.append(f"calibration frame {i}: counts must sum to n and have `bins` entries")
    if name == "regimes.json":
        g = len(obj["grid"])
        for i, f in enumerate(obj["frames"]):
            if len(f["ridges"]) != len(f["probs"]) or f["current"] >= len(f["probs"]):
                errs.append(f"regimes frame {i}: one ridge per regime probability")
            if any(len(r["density"]) != g for r in f["ridges"] + [f["unconditional"]]):
                errs.append(f"regimes frame {i}: densities must match the grid")
    if name == "exposure.json":
        n = len(obj["dates"])
        series = [obj["gross"], obj["net"]] + [obj["long"][c] for c in obj["classes"]] + [obj["short"][c] for c in obj["classes"]]
        if any(len(x) != n for x in series):
            errs.append("exposure: every series must have one value per date")
    return errs


def validate_dir(viz_dir=None, use_jsonschema=True):
    """Validate every file listed in the schema. Returns {file: [errors]}."""
    data = os.path.join(viz_dir or VIZ_DIR, "data")
    with open(os.path.join(data, "schema.json")) as fh:
        schema = json.load(fh)
    out = {}
    for name, sub in schema["files"].items():
        path = os.path.join(data, name)
        if not os.path.exists(path):
            out[name] = ["file missing"]
            continue
        with open(path) as fh:
            obj = json.load(fh)
        full = dict(sub, definitions=schema["definitions"])
        errs = []
        if use_jsonschema:
            try:
                import jsonschema
                errs += [f"{'.'.join(map(str, e.path))}: {e.message}"
                         for e in jsonschema.Draft7Validator(full).iter_errors(obj)]
            except ImportError:
                _check(obj, full, full, name, errs)
        else:
            _check(obj, full, full, name, errs)
        errs += consistency(name, obj)
        out[name] = errs
    return out
