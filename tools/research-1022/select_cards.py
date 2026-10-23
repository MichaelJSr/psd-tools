"""select_cards.py <repo> <base> <head> <trace.json>...: the cards a diff can change.

A card is selected when its trace (recorded at <base>) executed a line the
diff changes. Changed lines are widened to the whole statement around them
(a LINE event names a statement's first line), and an insertion selects the
statements on both sides of it. Comment-only and docstring-only hunks select
nothing, which is right: they cannot change a pixel."""
import ast, json, re, subprocess, sys

repo, base, head, *traces = sys.argv[1:]
diff = subprocess.run(["git", "-C", repo, "diff", "-U0", base, head, "--", "src/"], capture_output=True, text=True, check=True).stdout
changed = {}
cur = None
for line in diff.splitlines():
    if line.startswith("--- "):
        cur = None if line[4:] == "/dev/null" else line[6:]          # a/src/... -> src/...
        cur = cur[len("src/"):] if cur and cur.startswith("src/") else cur
    m = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line)
    if m and cur:
        a, b = int(m.group(1)), int(m.group(2) or 1)
        lines = range(a, a + b) if b else (a, a + 1)
        changed.setdefault(cur, set()).update(lines)

def statements(path_in_src, lines):
    """First lines of the innermost statements covering ``lines`` in the base file."""
    text = subprocess.run(["git", "-C", repo, "show", f"{base}:src/{path_in_src}"], capture_output=True, text=True, check=True).stdout
    tree = ast.parse(text)
    out = set()
    stmts = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
    for ln in lines:
        best = None
        for n in stmts:
            if n.lineno <= ln <= (n.end_lineno or n.lineno):
                # innermost: a compound statement's body lines belong to the inner statement
                if best is None or (n.end_lineno - n.lineno) < (best.end_lineno - best.lineno):
                    best = n
        if best is None:
            continue
        if isinstance(best, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ln != best.lineno:
            continue          # between a def's statements: a comment or blank line
        if isinstance(best, ast.Expr) and isinstance(best.value, ast.Constant) and isinstance(best.value.value, str):
            continue          # a docstring
        body = getattr(best, "body", None)
        if body and isinstance(body, list) and ln < body[0].lineno:
            out.update(range(best.lineno, body[0].lineno))      # a compound statement's header
        else:
            out.update(range(best.lineno, (best.end_lineno or best.lineno) + 1))
    return out

hot = {f: statements(f, ls) for f, ls in changed.items() if f.endswith(".py")}
print("changed statements:", {f: len(v) for f, v in hot.items()})
for t in traces:
    d = json.load(open(t))
    hit = sorted({(f, l) for f, ls in d["lines"].items() if f in hot for l in ls if l in hot[f]})
    name = d["psd"].rsplit("/", 1)[1]
    print(("SELECT " if hit else "skip   ") + name, hit[:4])
