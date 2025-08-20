#!/bin/bash

# 1️⃣ Optional: remove old kernels
echo "Cleaning up old Jupyter kernels..."
for K in $(jupyter kernelspec list --json | jq -r '.kernelspecs | keys[]'); do
    if [[ "$K" == "project-venv" ]] || [[ "$K" == "global-env" ]]; then
        jupyter kernelspec uninstall -f $K
        echo "Removed old kernel: $K"
    fi
done

# 2️⃣ Create global venv
GLOBAL_VENV="$HOME/pyenv_global"
if [ ! -d "$GLOBAL_VENV" ]; then
    echo "Creating global virtual environment at $GLOBAL_VENV"
    python3 -m venv "$GLOBAL_VENV"
else
    echo "Global virtual environment already exists at $GLOBAL_VENV"
fi

# 3️⃣ Activate venv
source "$GLOBAL_VENV/bin/activate"

# 4️⃣ Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# 5️⃣ Install main packages
echo "Installing packages..."
pip install streamlit jupyter ipykernel pandas numpy pyarrow altair pydeck

# 6️⃣ Register venv as Jupyter kernel
echo "Registering global kernel for Jupyter..."
python -m ipykernel install --user --name=global-env --display-name "Python (global-env)"

# 7️⃣ Show instructions for VS Code
echo "-------------------------------------"
echo "Setup complete!"
echo "Global venv path for VS Code interpreter:"
echo "$GLOBAL_VENV/bin/python"
echo "In VS Code:"
echo "  - Python: Select Interpreter → Enter interpreter path → $GLOBAL_VENV/bin/python"
echo "  - Jupyter notebooks: Select Kernel → Python (global-env)"
echo "-------------------------------------"
