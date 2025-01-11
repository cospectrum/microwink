import pytest
import os
from pathlib import Path


Script = str


@pytest.mark.parametrize(["path"], [("./README.md",)])
def test_readme(path: os.PathLike) -> None:
    path = Path(path)
    if not path.exists():
        pytest.skip(reason=f"{path} doesn't exist")

    text = path.read_text()
    scripts = parse_md_scripts(text, "```python\n", "```\n")
    scripts += parse_md_scripts(text, "```py\n", "```\n")
    if len(scripts) == 0:
        pytest.skip(reason=f"no scripts in {path}")

    for script in scripts:
        print("\n# executing the following script")
        print(script)
        print("\n# stdout...")
        exec(script)


def parse_md_scripts(text: str, start_tag: str, end_tag) -> list[Script]:
    _, *sections = text.split(start_tag)
    scripts = []
    for section in sections:
        script, *_ = section.split(end_tag)
        scripts.append(script.strip())
    return scripts
