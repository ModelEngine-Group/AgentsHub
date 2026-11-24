"""Compare two OpenAPI contracts without contacting the described API."""

import argparse
import json
from pathlib import Path
import re
from urllib.parse import unquote


METHODS = {"get", "put", "post", "delete", "options", "head", "patch", "trace"}
COMPLEX_SCHEMA = {"oneOf", "anyOf", "allOf", "not", "if", "then", "else", "$dynamicRef"}


def parse(text):
    if len(text) > 250_000:
        raise ValueError("Each OpenAPI document must be at most 250,000 characters.")
    doc = json.loads(text)
    if not isinstance(doc, dict) or not re.fullmatch(r"3\.[01]\.\d+", str(doc.get("openapi"))):
        raise ValueError("Provide an OpenAPI 3.0 or 3.1 document encoded as JSON.")
    if not isinstance(doc.get("paths"), dict):
        raise ValueError("The paths field must be an object.")
    return doc


def pointer(*parts):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def resolve(doc, value):
    seen = set()
    while isinstance(value, dict) and "$ref" in value:
        ref = value["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/") or ref in seen or len(seen) >= 20:
            raise ValueError("A reference is external, recursive, or too deep.")
        seen.add(ref)
        value = doc
        for token in unquote(ref[2:]).split("/"):
            value = value[token.replace("~1", "/").replace("~0", "~")]
    if not isinstance(value, dict):
        raise ValueError("The referenced item is not a supported object.")
    return value


def review_openapi(before: str, after: str) -> dict:
    """Compare OpenAPI 3.0/3.1 JSON. Return evidence, severity and contract locations."""
    left, right = parse(before), parse(after)
    findings = []

    def add(level, code, path, old, new):
        findings.append({"level": level, "code": code, "path": pointer(*path), "before": old, "after": new})

    def pair(old, new, path):
        try:
            return resolve(left, old), resolve(right, new)
        except (ValueError, KeyError, TypeError):
            add("review", "reference-or-structure", path, None, None)
            return {}, {}

    def compare_schema(old, new, path, direction, depth=0):
        old, new = pair(old, new, path)
        if depth >= 6 or COMPLEX_SCHEMA.intersection(old.keys() | new.keys()):
            add("review", "complex-schema", path, None, None)
            return
        old_type, new_type = old.get("type"), new.get("type")
        if old_type != new_type:
            add("breaking" if isinstance(old_type, str) and isinstance(new_type, str) else "review",
                "type-changed", (*path, "type"), old_type, new_type)
        old_enum, new_enum = old.get("enum"), new.get("enum")
        if old_enum != new_enum:
            if isinstance(old_enum, list) and isinstance(new_enum, list):
                lost = [value for value in old_enum if value not in new_enum]
                added = [value for value in new_enum if value not in old_enum]
                if direction == "input" and lost:
                    add("breaking", "request-enum-narrowed", (*path, "enum"), old_enum, new_enum)
                if direction == "output" and added:
                    add("review", "response-enum-expanded", (*path, "enum"), old_enum, new_enum)
            else:
                add("review", "enum-constraint-changed", (*path, "enum"), old_enum, new_enum)
        for field in ("format", "nullable", "minimum", "maximum", "minLength", "maxLength",
                      "pattern", "additionalProperties", "default"):
            if old.get(field) != new.get(field):
                add("review", "schema-constraint-changed", (*path, field), old.get(field), new.get(field))
        ignored_property = "readOnly" if direction == "input" else "writeOnly"
        old_props = {name: value for name, value in old.get("properties", {}).items()
                     if not (isinstance(value, dict) and value.get(ignored_property))}
        new_props = {name: value for name, value in new.get("properties", {}).items()
                     if not (isinstance(value, dict) and value.get(ignored_property))}
        old_required = set(old.get("required", [])) & old_props.keys()
        new_required = set(new.get("required", [])) & new_props.keys()
        for name in sorted(new_required - old_required if direction == "input" else old_required - new_required):
            add("breaking", "request-property-required" if direction == "input" else "response-guarantee-removed",
                (*path, "properties", name), name in old_required, name in new_required)
        for name in sorted(old_props.keys() - new_props.keys()):
            add("breaking" if direction == "output" else "review", "property-removed",
                (*path, "properties", name), True, False)
        for name in sorted(old_props.keys() & new_props.keys()):
            compare_schema(old_props[name], new_props[name], (*path, "properties", name), direction, depth + 1)
        if "items" in old and "items" in new:
            compare_schema(old["items"], new["items"], (*path, "items"), direction, depth + 1)

    def parameters(doc, item, operation, path):
        result = {}
        for owner, scope in ((item, path[:-1]), (operation, path)):
            for index, raw in enumerate(owner.get("parameters", [])):
                location = (*scope, "parameters", index)
                try:
                    param = resolve(doc, raw)
                    result[(param["in"], param["name"])] = (param, location)
                except (ValueError, KeyError, TypeError):
                    add("review", "parameter-reference", location, None, None)
        return result

    def compare_content(old, new, path, direction):
        for media in sorted(old.keys() - new.keys()):
            add("breaking", "media-type-removed", (*path, media), True, False)
        for media in sorted(old.keys() & new.keys()):
            compare_schema(old[media].get("schema", {}), new[media].get("schema", {}),
                           (*path, media, "schema"), direction)

    for route, old_item in sorted(left["paths"].items()):
        if not route.startswith("/"):
            continue
        new_item = right["paths"].get(route, {})
        old_item, new_item = pair(old_item, new_item, ("paths", route))
        for method in sorted(METHODS & old_item.keys()):
            path = ("paths", route, method)
            if method not in new_item:
                add("breaking", "operation-removed", path, True, False)
                continue
            old, new = old_item[method], new_item[method]
            old_params, new_params = parameters(left, old_item, old, path), parameters(right, new_item, new, path)
            for key, (param, location) in sorted(new_params.items()):
                prior, _ = old_params.get(key, ({}, None))
                if param.get("required") and not prior.get("required"):
                    add("breaking", "request-parameter-required", (*location, "required"), False, True)
                if key in old_params:
                    compare_schema(prior.get("schema", {}), param.get("schema", {}), (*location, "schema"), "input")
                    for field in ("style", "explode", "allowReserved", "content"):
                        if prior.get(field) != param.get(field):
                            add("review", "parameter-serialization", (*location, field), None, None)
            for key in sorted(old_params.keys() - new_params.keys()):
                _, location = old_params[key]
                add("review", "request-parameter-removed", location, True, False)
            old_body, new_body = pair(old.get("requestBody", {}), new.get("requestBody", {}), (*path, "requestBody"))
            if new_body.get("required") and not old_body.get("required"):
                add("breaking", "request-body-required", (*path, "requestBody", "required"), False, True)
            compare_content(old_body.get("content", {}), new_body.get("content", {}), (*path, "requestBody", "content"), "input")
            new_responses = new.get("responses", {})
            for status, prior in sorted(old.get("responses", {}).items()):
                if not re.fullmatch(r"2(?:\d\d|XX)", status):
                    continue
                response_path = (*path, "responses", status)
                if status not in new_responses:
                    add("review", "success-status-removed", response_path, True, False)
                    continue
                old_response, new_response = pair(prior, new_responses[status], response_path)
                compare_content(old_response.get("content", {}), new_response.get("content", {}), (*response_path, "content"), "output")
            if old.get("security", left.get("security", [])) != new.get("security", right.get("security", [])):
                add("review", "security-changed", (*path, "security"), None, None)
            if old.get("operationId") != new.get("operationId"):
                add("review", "operation-id-changed", (*path, "operationId"), old.get("operationId"), new.get("operationId"))
    findings.sort(key=lambda item: (item["level"] != "breaking", item["path"], item["code"]))
    return {"count": len(findings), "findings": findings[:50], "truncated": len(findings) > 50}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, help="Read two UTF-8 JSON contracts locally.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.files:
        if len(args.files) != 2:
            parser.error("Provide before.json and after.json.")
        result = review_openapi(*(path.read_text(encoding="utf-8-sig") for path in args.files))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    from fastmcp import FastMCP

    mcp = FastMCP("openapi-contract-review")
    mcp.tool(review_openapi)
    mcp.run(transport="streamable-http", host=args.host, port=args.port, stateless_http=True)


if __name__ == "__main__":
    main()
