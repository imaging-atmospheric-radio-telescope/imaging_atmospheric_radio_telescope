import os
import tarfile
import shutil
import subprocess
import glob
import hashlib


CORSIKA_77100_TAR_GZ_HASH_HEXDIGEST = "bc67b14c957a024baf7f2893ab246d34"
CORSIKA_DOWNLOAD_URL = "https://web.ikp.kit.edu/corsika/download/old/v771/"
CORSIKA_NAME = "corsika-77100"
CORSIKA_TAR_FILENAME = CORSIKA_NAME + ".tar.gz"


def md5sum(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def call_and_save_std(target, o_path, e_path, stdin=None):
    with open(o_path, "w") as stdout, open(e_path, "w") as stderr:
        subprocess.call(target, stdout=stdout, stderr=stderr, stdin=stdin)


def download(
    output_dir, username, password, corsika_download_url, corsika_tar_filename,
):
    # download CORSIKA from KIT
    # wget uses $http_proxy environment-variables in case of proxy
    call_and_save_std(
        target=[
            "wget",
            "--no-check-certificate",
            "--directory-prefix",
            output_dir,
            "--user",
            username,
            "--password",
            password,
            corsika_download_url + corsika_tar_filename,
        ],
        stdout_path=os.path.join(
            output_dir, corsika_tar_filename + ".wget.stdout"
        ),
        stderr_path=os.path.join(
            output_dir, corsika_tar_filename + ".wget.stderr"
        ),
    )


def install(corsika_tar_path, corsika_config_path, install_path):
    install_path = os.path.abspath(install_path)
    corsika_config_path = os.path.abspath(corsika_config_path)

    os.makedirs(install_path, exist_ok=True)

    # untar, unzip the CORSIKA download
    tar = tarfile.open(corsika_tar_path)
    tar.extractall(path=install_path)
    tar.close()

    # Go into CORSIKA dir
    corsika_basename = os.path.basename(
        os.path.splitext(os.path.splitext(corsika_tar_path)[0])[0]
    )
    corsika_path = os.path.join(install_path, corsika_basename)
    os.chdir(corsika_path)

    # Provide the coconut config.h
    shutil.copyfile(corsika_config_path, os.path.join("include", "config.h"))

    # coconut configure
    call_and_save_std(
        target=["./coconut"],
        o_path="../coconut_configure.o",
        e_path="../coconut_configure.e",
        stdin=open("/dev/null", "r"),
    )

    # coconut build
    call_and_save_std(
        target=["./coconut", "-i"],
        o_path="../coconut_make.o",
        e_path="../coconut_make.e",
    )

    # Copy std ATMPROFS to the CORSIKA run directory
    for atmprof in glob.glob("bernlohr/atmprof*"):
        shutil.copy(atmprof, "run")

    assert os.path.isfile(
        os.path.join("run", "corsika77100Linux_QGSII_urqmd_coreas")
    )
