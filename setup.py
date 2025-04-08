from setuptools import setup
from pathlib import Path
import shutil

# -----------------------------
# Paths & File Copy for Scripts
# -----------------------------
project_root = Path(__file__).parent.resolve()
run_src = project_root / "run.py"
run_mp_src = project_root / "run_mp.py"
chemts_dir = project_root / "chemtsv2"

# Ensure chemtsv2/ exists
chemts_dir.mkdir(exist_ok=True)

# Copy CLI entry scripts into package directory
if run_src.exists():
    shutil.copy(run_src, chemts_dir / "run.py")
if run_mp_src.exists():
    shutil.copy(run_mp_src, chemts_dir / "run_mp.py")

# -----------------------------
# Metadata
# -----------------------------
DOCLINES = (__doc__ or '').splitlines()
DESCRIPTION = DOCLINES[0] if DOCLINES else "ChemTSv2: Molecular Generator via MCTS"
LONG_DESCRIPTION = '\n'.join(DOCLINES[2:]) if len(DOCLINES) > 2 else ""

# -----------------------------
# Setup configuration
# -----------------------------
setup(
    name="chemtsv2",
    version="0.9.10",
    author="Shoichi Ishida",
    author_email="ishida.sho.nm@yokohama-cu.ac.jp",
    maintainer="Shoichi Ishida",
    maintainer_email="ishida.sho.nm@yokohama-cu.ac.jp",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="MIT License",
    url="https://github.com/molecule-generator-collection/ChemTSv2",
    download_url="https://github.com/molecule-generator-collection/ChemTSv2",
    python_requires=">=3.7",
    install_requires=[
        'tensorflow==2.5',
        'numpy~=1.19.2',
        'protobuf~=3.9.2',
        'rdkit-pypi==2021.03.5',
        'pyyaml',
        'pandas',
        'joblib'
    ],
    packages=[
        'chemtsv2',
        'chemtsv2.misc'
    ],
    entry_points={
        'console_scripts': [
            "chemtsv2 = chemtsv2.run:main",
            "chemtsv2-mp = chemtsv2.run_mp:main"
        ]
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
