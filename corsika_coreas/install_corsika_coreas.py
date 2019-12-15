#!/usr/bin/env python
# Copyright 2017-2019 Sebastian A. Mueller
"""
Install the CORSIKA for the Askarian-telescope.

Usage: install_corsika_coreas -p=PATH --username=USERNAME --password=PASSWORD

Options:
    -p --install_path=PATH              The path to install CORSIKA in
    --username=USERNAME                 Username for the KIT-CORSIKA-server
    --password=PASSWORD                 Password fot the KIT-CORSIKA-server

During the installation, the stdout and stderr of 'coconut_configure'
and 'coconut_make' are written to the install_path.
Visit the CORSIKA homepage: https://www.ikp.kit.edu/corsika/
to test your username and password in the download section.
If you do not have yet the CORSIKA username and password, drop the developers
of CORSIKA an e-mail and kindly ask for it.
"""
import docopt
import sys
import os
import tarfile
import shutil
import glob
import subprocess


def call_and_save_std(target, o_path, e_path, stdin=None):
    with open(o_path, 'w') as stdout, open(e_path, 'w') as stderr:
        subprocess.call(target, stdout=stdout, stderr=stderr, stdin=stdin)


def install(install_path, username, password):
    corsika_config_path = os.path.abspath('./config.h')
    assert os.path.isfile(corsika_config_path)

    os.makedirs(install_path, exist_ok=True)
    os.chdir(install_path)

    # download CORSIKA from KIT
    web_path = 'https://web.ikp.kit.edu/corsika/download/corsika-v770/'
    corsika_tarname = 'corsika-77100.tar.gz'
    link = web_path + corsika_tarname
    call_and_save_std(
        target=['wget', '--user', username, '--password', password, link],
        o_path='wget.o',
        e_path='wget.e')

    # untar, unzip the CORSIKA download
    tar = tarfile.open(corsika_tarname)
    tar.extractall(path='.')
    tar.close()

    # Go into CORSIKA dir
    corsika_path = os.path.splitext(os.path.splitext(corsika_tarname)[0])[0]
    os.chdir(corsika_path)

    # Provide the Askarian Telescope coconut config.h
    shutil.copyfile(corsika_config_path, 'include/config.h')

    # coconut configure
    call_and_save_std(
        target=['./coconut'],
        o_path='../coconut_configure.o',
        e_path='../coconut_configure.e',
        stdin=open('/dev/null', 'r'))

    # coconut build
    call_and_save_std(
        target=['./coconut', '-i'],
        o_path='../coconut_make.o',
        e_path='../coconut_make.e')

    # Copy std ATMPROFS to the CORSIKA run directory
    for atmprof in glob.glob('bernlohr/atmprof*'):
        shutil.copy(atmprof, 'run')

    if os.path.isfile('run/corsika77100Linux_QGSII_urqmd_coreas'):
        sys.exit(0)
    else:
        sys.exit(1)


def main():
    try:
        arguments = docopt.docopt(__doc__)
        install(
            install_path=arguments['--install_path'],
            username=arguments['--username'],
            password=arguments['--password'])
    except docopt.DocoptExit as e:
        print(e)

if __name__ == '__main__':
    main()
