"""Global seed control and run provenance.

Every reported number has to be reproducible by someone who was not there. In
practice that means two things, and this module does both.

**Seed everything, in one call.** There are four independent random sources in
this pipeline -- Python's ``random``, NumPy, PyTorch (CPU and CUDA), and the
``PYTHONHASHSEED`` that governs set and dict iteration order. Seeding three of
four is the usual failure: results move between runs and the cause is invisible
because the unseeded source is not the one anybody is thinking about.

**Record what actually ran.** A seed alone does not reproduce a number if the
model checkpoint, the hardware profile, the contract version or the tier map
has changed underneath it. :func:`run_manifest` captures those alongside the
git commit, so a result in a paper can be traced to the state that produced it.

Usage::

    import reproducibility
    reproducibility.seed_everything(0)
    print(reproducibility.run_manifest())
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_SEED = 0


def seed_everything(seed: int = DEFAULT_SEED, *, deterministic_torch: bool = True) -> int:
    """Seed every random source this pipeline touches. Returns the seed.

    ``deterministic_torch`` also disables cuDNN's autotuner. That costs some
    throughput and is the right trade for reported numbers: the autotuner picks
    different convolution algorithms on different runs, and those can change
    results in the last decimal place -- enough to make a table irreproducible
    while looking like noise.

    Note that ``PYTHONHASHSEED`` only takes effect at interpreter startup, so
    setting it here helps subprocesses, not the current process. For a fully
    pinned run, export it before invoking Python. :func:`hash_seed_is_pinned`
    reports whether that happened.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def hash_seed_is_pinned() -> bool:
    """Whether ``PYTHONHASHSEED`` was set before the interpreter started.

    Set afterwards it is decorative: string hashing, and therefore set and dict
    iteration order, was already fixed for this process.
    """
    return os.environ.get("PYTHONHASHSEED") is not None and "PYTHONHASHSEED" in os.environ


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).resolve().parent)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


@dataclass
class RunManifest:
    """Everything needed to reproduce a reported number."""

    seed: int
    git_commit: str | None
    git_dirty: bool
    python: str
    platform: str
    torch: str | None
    cuda_device: str | None
    contract_version: str
    hardware_profile: str
    model_tiers: dict[str, str | None] = field(default_factory=dict)
    thresholds_tuned: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Short hash of the manifest. Two runs sharing it are comparable."""
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def render(self) -> str:
        lines = [
            "Run provenance",
            "-" * 64,
            f"  fingerprint       {self.fingerprint()}",
            f"  seed              {self.seed}",
            f"  git commit        {self.git_commit or '(not a repo)'}"
            + ("  DIRTY" if self.git_dirty else ""),
            f"  contract          v{self.contract_version}",
            f"  hardware profile  {self.hardware_profile}",
            f"  python            {self.python}",
            f"  torch             {self.torch or '(not installed)'}",
            f"  device            {self.cuda_device or 'cpu'}",
        ]
        if self.model_tiers:
            lines.append("  model tiers:")
            for tier, model in sorted(self.model_tiers.items()):
                lines.append(f"    {tier:<18} {model or '(unset)'}")
        if not self.thresholds_tuned:
            lines.append("  NOTE: the escalation band is a PLACEHOLDER, not tuned on data.")
        if self.git_dirty:
            lines.append("  WARNING: uncommitted changes. This run is not reproducible "
                         "from the commit alone.")
        return "\n".join(lines)


def run_manifest(seed: int = DEFAULT_SEED, **extra: Any) -> RunManifest:
    """Capture the state a reported number depends on."""
    torch_version = cuda_device = None
    try:
        import torch

        torch_version = torch.__version__
        if torch.cuda.is_available():
            cuda_device = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    import settings
    from contract.models import CONTRACT_VERSION

    try:
        hw = settings.hardware()
        profile, tuned = hw.name, hw.thresholds_tuned
    except Exception:
        profile, tuned = "(unreadable)", False

    tiers: dict[str, str | None] = {}
    try:
        cfg = settings.models()
        tiers = {name: spec.model for name, spec in cfg.tiers.items()}
    except Exception:
        pass

    status = _git("status", "--porcelain")
    return RunManifest(
        seed=seed,
        git_commit=_git("rev-parse", "--short", "HEAD"),
        git_dirty=bool(status),
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()}",
        torch=torch_version,
        cuda_device=cuda_device,
        contract_version=CONTRACT_VERSION,
        hardware_profile=profile,
        model_tiers=tiers,
        thresholds_tuned=tuned,
        extra=extra,
    )


def start_run(seed: int = DEFAULT_SEED, *, quiet: bool = False, **extra: Any) -> RunManifest:
    """Seed everything and return the manifest. Call at the top of any runner."""
    seed_everything(seed)
    manifest = run_manifest(seed, **extra)
    if not quiet:
        print(manifest.render())
        print()
    return manifest
