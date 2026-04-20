from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_architecture_doc_contains_mermaid_diagrams():
    content = (PROJECT_ROOT / "docs/system_architecture.md").read_text(encoding="utf-8")

    assert content.count("```mermaid") >= 2
    assert "CT Trigger" in content
    assert "Prometheus" in content


def test_demo_runbook_references_scripts():
    content = (PROJECT_ROOT / "docs/demo_runbook.md").read_text(encoding="utf-8")

    assert "scripts/demo-prepare.sh" in content
    assert "scripts/demo-warmup.sh" in content
    assert "scripts/demo-open-tabs.sh" in content
    assert "scripts/demo-stop.sh" in content


def test_readme_points_to_current_streamlit_entrypoint_and_demo_docs():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "streamlit run src/streamlit.py" in content
    assert "docs/system_architecture.md" in content
    assert "docs/demo_runbook.md" in content


def test_demo_scripts_have_valid_shell_syntax():
    for script_name in [
        "scripts/demo-prepare.sh",
        "scripts/demo-warmup.sh",
        "scripts/demo-open-tabs.sh",
        "scripts/demo-stop.sh",
    ]:
        result = subprocess.run(
            ["bash", "-n", str(PROJECT_ROOT / script_name)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
