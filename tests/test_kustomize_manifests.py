import shutil
import subprocess


def test_kubernetes_kustomize_bundle_builds():
    kubectl = shutil.which("kubectl")
    assert kubectl is not None, "kubectl client is required to validate kustomize output"

    result = subprocess.run(
        [kubectl, "kustomize", "deployment/kubernetes"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "kind: Deployment" in result.stdout
    assert "kind: Service" in result.stdout


def test_monitoring_kustomize_bundle_builds():
    kubectl = shutil.which("kubectl")
    assert kubectl is not None, "kubectl client is required to validate kustomize output"

    result = subprocess.run(
        [kubectl, "kustomize", "deployment/monitoring"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "kind: ServiceMonitor" in result.stdout
    assert "kind: PrometheusRule" in result.stdout
    assert "grafana_dashboard: \"1\"" in result.stdout
