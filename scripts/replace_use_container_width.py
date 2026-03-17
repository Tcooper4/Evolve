from pathlib import Path
import ast


def main() -> None:
    changed = []
    files = list(Path("pages").rglob("*.py")) + list(
        Path("trading").rglob("*.py")
    ) + list(Path("components").rglob("*.py"))

    for p in files:
        src = p.read_text(encoding="utf-8", errors="replace")
        new = (
            src.replace("use_container_width=True", "width='stretch'")
            .replace("use_container_width=False", "width='content'")
        )
        if new != src:
            try:
                ast.parse(new)
            except SyntaxError:
                print(f"SKIP (syntax) {p}")
                continue
            p.write_text(new, encoding="utf-8")
            changed.append(str(p))

    for f in changed:
        print(f"Fixed: {f}")
    print(f"Total: {len(changed)} files")


if __name__ == "__main__":
    main()

