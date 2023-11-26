import subprocess

print(subprocess.run(["sh 0_deploy_prerequisites/download_reqs_set_vars.sh"], shell=True))