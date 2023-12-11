import subprocess

print(subprocess.run(["sh 1_session-setup-chroma-db/setup-chroma.sh"], shell=True))