from pathlib import Path


def test_install_oeq_script_exports_gcc14_workaround():
    script = Path("scripts/install_oeq.sh").read_text()
    assert "set -euo pipefail" in script
    assert "CXXFLAGS=\"${CXXFLAGS:-} -include cstdint\"" in script
    assert "pip install" in script
    assert "openequivariance" in script
