"""Quick API smoke test using urllib (no external deps)."""
import urllib.request
import urllib.parse
import json

BASE = "http://127.0.0.1:8000"


def api_get(path):
    with urllib.request.urlopen(f"{BASE}{path}") as r:
        return json.loads(r.read())


def api_post(path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def pp(label, data):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print("─" * 50)
    print(json.dumps(data, indent=2, default=str))


# 1. Sections v1
secs = api_get("/sections?version=1")
pp("TOP-LEVEL SECTIONS (v1)", secs)

# 2. Search
search = api_get("/search?q=overpressure&version=1")
pp("SEARCH: overpressure", {"count": search["count"], "results": [
    {"id": r["id"], "stable_path": r["stable_path"], "heading": r["heading"]}
    for r in search["results"]
]})

# 3. Single node detail
first_id = secs["sections"][0]["id"]
node = api_get(f"/sections/{first_id}")
pp(f"NODE DETAIL id={first_id}", {
    "heading": node["heading"],
    "stable_path": node["stable_path"],
    "content_hash": node["content_hash"],
    "children_count": len(node.get("children", [])),
})

# 4. Diff v1 vs v2
diff = api_get("/versions/1/diff/2")
pp("DIFF v1 vs v2", {
    "unchanged_count": diff["unchanged_count"],
    "modified_count": len(diff["modified"]),
    "modified_paths": [m["stable_path"] for m in diff["modified"]],
    "added": diff["added"],
    "deleted": diff["deleted"],
})

# 5. Create selection (use section 4.1, 4.2 — safety-critical content)
ov_q = urllib.parse.urlencode({"q": "overpressure", "version": 1})
all_nodes = api_get(f"/search?{ov_q}")
ov_node = all_nodes["results"][0] if all_nodes["results"] else None

ec_q = urllib.parse.urlencode({"q": "error codes", "version": 1})
ec_nodes = api_get(f"/search?{ec_q}")
ec_node = ec_nodes["results"][0] if ec_nodes["results"] else None

if ov_node and ec_node:
    sel = api_post("/selections", {
        "name": "Safety-critical: Overpressure + Error Codes",
        "node_ids": [ov_node["id"], ec_node["id"]],
        "version_id": 1,
    })
    pp("SELECTION CREATED", sel)
    print(f"\n\u2705 Selection ID: {sel['id']}")
    print(f"   Nodes: {[n['stable_path'] for n in sel['nodes']]}")
else:
    print("WARNING: Could not find safety-critical nodes by search")
