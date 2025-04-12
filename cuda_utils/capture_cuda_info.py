import subprocess
import sys
import os

# Path to the virtual environment Python
venv_python = os.path.join("Z:", "Coding", "jarvis_v1", ".venv", "Scripts", "python.exe")
check_script = os.path.join("Z:", "Coding", "jarvis_v1", "check_cuda.py")

# Run the check_cuda.py script and capture output
result = subprocess.run([venv_python, check_script],
                        capture_output=True,
                        text=True)

# Write the output to a file
with open("cuda_check_results.txt", "w") as f:
    f.write("STDOUT:\n")
    f.write(result.stdout)

    if result.stderr:
        f.write("\n\nSTDERR:\n")
        f.write(result.stderr)

print(f"CUDA check results written to cuda_check_results.txt")
