import subprocess

def call_and_save_std(target, o_path, e_path, stdin=None):
    with open(o_path, "w") as stdout, open(e_path, "w") as stderr:
        subprocess.call(target, stdout=stdout, stderr=stderr, stdin=stdin)
