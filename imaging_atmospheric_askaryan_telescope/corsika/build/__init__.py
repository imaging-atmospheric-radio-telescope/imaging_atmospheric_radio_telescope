import os
import tarfile
import shutil
import glob
import hashlib
import pkg_resources
from . import utils


CORSIKA_77100_TAR_GZ_MD5SUM = "bc67b14c957a024baf7f2893ab246d34"
CORSIKA_DOWNLOAD_URL = "https://web.ikp.kit.edu/corsika/download/old/v771/"
CORSIKA_NAME = "corsika-77100"
CORSIKA_TAR_FILENAME = CORSIKA_NAME + ".tar.gz"


def get_corsika_executable_path():
    build_dir = get_corsika_build_path()
    return os.path.join(
        build_dir, CORSIKA_NAME, "run", "corsika77100Linux_QGSII_urqmd_coreas"
    )


def get_corsika_build_path():
    """
    Returns the default build directory path for CORSIKA CoREAS used by this
    python package.
    """
    return pkg_resources.resource_filename(
        "imaging_atmospheric_askaryan_telescope",
        os.path.join("corsika", "__build__"),
    )


def get_corsika_config_path():
    """
    Returns the path to the config.h header file used by CORSIKA to controll
    what features of CORSIKA will be build.
    The header is used by CORSIKA's 'coconut' build script.
    """
    return pkg_resources.resource_filename(
        "imaging_atmospheric_askaryan_telescope",
        os.path.join("corsika", "build", "resources", "config.h"),
    )


def md5sum(path):
    """
    Returns the md5 checksum of the file in 'path'.
    Computation is done block, by block to reduce memory load.
    """
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(
    output_dir,
    username,
    password,
    corsika_download_url=CORSIKA_DOWNLOAD_URL,
    corsika_tar_filename=CORSIKA_TAR_FILENAME,
):
    """
    Downloads CORSIKA.tar from KIT's servers.
    Kindly ask the CORSIKA developers to provide you the username/password.

    Parameters
    ----------
    output_dir : str
        Directory to save CORSIKA.tar in.
    username : str
        The CORSIKA username for downloads.
    password : str
        The CORSIKA password for downloads.
    corsika_download_url : str, default: CORSIKA_DOWNLOAD_URL
        URL to the directory to download from.
    corsika_tar_filename : str, default: CORSIKA_TAR_FILENAME
        The filename to download

    Internal
    --------
    Uses wget.
    wget uses $http_proxy environment-variables in case of proxy
    """
    utils.call_and_save_std(
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


def uninstall(build_dir=None):
    """
    Uninstall and removes the CORSIKA CoREAS build.
    """
    join = os.path.join
    print("Uninstall CORSIKA CoREAS ...")

    if build_dir is None:
        build_dir = get_corsika_build_path()
    build_dir = os.path.abspath(build_dir)
    print(f"build_dir: '{build_dir:s}'")

    os.remove(join(build_dir, "coconut_configure.o"))
    os.remove(join(build_dir, "coconut_configure.e"))

    os.remove(join(build_dir, "coconut_make.o"))
    os.remove(join(build_dir, "coconut_make.e"))

    if os.path.isdir(join(build_dir, CORSIKA_NAME)):
        shutil.rmtree(join(build_dir, CORSIKA_NAME))
    print("Uninstall CORSIKA CoREAS ... Done")


def install(build_dir=None, corsika_tar_gz_path=None, corsika_config_path=None):
    """
    Install CORSIKA CoREAS for the
    'imaging atmospheric askaryan telescope' package.

    Parameters
    ----------
    build_dir : str (default: None)
        Directory to build in. If None, 'buikd_dir' will be in this pyhton
        package's install path.
    corsika_tar_gz_path : str (default: None)
        Path to the corsika.tar which you downloaded from KIT. If None, the
        corsika.tar is expected to be in the 'build_dir'.
    corsika_config_path : str (default: None)
        Path to the config.h header-file which exactly defines how
        to build this flavor of corsika. If None, the path to the default config
        of this pyhton package is used.
    """
    print("Building CORSIKA CoREAS ...")

    if corsika_config_path is None:
        corsika_config_path = get_corsika_config_path()
    corsika_config_path = os.path.abspath(corsika_config_path)
    print(f"corsika_config_path: '{corsika_config_path:s}'")

    if build_dir is None:
        build_dir = get_corsika_build_path()
    build_dir = os.path.abspath(build_dir)
    print(f"build_dir: '{build_dir:s}'")

    if corsika_tar_gz_path is None:
        corsika_tar_gz_path = os.path.join(build_dir, CORSIKA_TAR_FILENAME)
    corsika_tar_gz_path = os.path.abspath(corsika_tar_gz_path)
    print(f"corsika_tar_gz_path: '{corsika_tar_gz_path:s}'")

    version_good = is_expected_version(corsika_tar_gz_path=corsika_tar_gz_path)
    if version_good:
        print("corsika_tar_gz md5sum: OK.")
    else:
        print(
            f"corsika_tar_gz md5sum: BAD. "
            f"Expected '{CORSIKA_77100_TAR_GZ_MD5SUM:s}'."
        )

    os.makedirs(build_dir, exist_ok=True)

    print("untar, unzip the corsika_tar_gz.")
    tar = tarfile.open(corsika_tar_gz_path)
    tar.extractall(path=build_dir)
    tar.close()

    # Go into CORSIKA dir
    corsika_basename = os.path.basename(
        os.path.splitext(os.path.splitext(corsika_tar_gz_path)[0])[0]
    )
    corsika_path = os.path.join(build_dir, corsika_basename)
    os.chdir(corsika_path)

    print("Provide the coconut config.h")
    shutil.copyfile(corsika_config_path, os.path.join("include", "config.h"))

    print("Calling 'coconut configure'")
    utils.call_and_save_std(
        target=["./coconut"],
        o_path="../coconut_configure.o",
        e_path="../coconut_configure.e",
        stdin=open("/dev/null", "r"),
    )

    print("Calling 'coconut -i' to compile")
    utils.call_and_save_std(
        target=["./coconut", "-i"],
        o_path="../coconut_make.o",
        e_path="../coconut_make.e",
    )

    print("Copy ATMPROFS to the CORSIKA run directory")
    for atmprof in glob.glob("bernlohr/atmprof*"):
        shutil.copy(atmprof, "run")


    if os.path.exists(get_corsika_executable_path()):
        print("Building CORSIKA CoREAS ... SUCCESS.")
    else:
        print("Building CORSIKA CoREAS ... FAIL.")

def is_expected_version(corsika_tar_gz_path):
    """
    Returns True/False depending on 'corsika_tar_gz_path' has the
    expected md5sum.
    """
    actual_md5sum = md5sum(path=corsika_tar_gz_path)
    if actual_md5sum == CORSIKA_77100_TAR_GZ_MD5SUM:
        return True
    else:
        msg = ""
        msg += "Expected '{:s}' ".format(corsika_tar_gz_path)
        msg += "to have md5sum: '{:s}' of '{:s}', ".format(
            CORSIKA_77100_TAR_GZ_MD5SUM,
            CORSIKA_TAR_FILENAME,
        )
        msg += "but it actually has md5sum: {:s}.".format(actual_md5sum)
        print(msg)
        return False
