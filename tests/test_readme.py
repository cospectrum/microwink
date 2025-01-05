from pathlib import Path


Script = str


def test_readme() -> None:
    readme_path = Path("./README.md")
    scripts = parse_readme_code(readme_path, "```python\n", "```\n")
    scripts += parse_readme_code(readme_path, "```py\n", "```\n")
    assert len(scripts) > 0
    for script in scripts:
        print("\n# executing the following script")
        print(script)
        print("\n# stdout...")
        exec(script)


def parse_readme_code(path: Path, start_tag: str, end_tag) -> list[Script]:
    assert path.exists()
    text = path.read_text()
    _, *sections = text.split(start_tag)
    scripts = []
    for section in sections:
        script, *_ = section.split(end_tag)
        scripts.append(script.strip())
    return scripts
