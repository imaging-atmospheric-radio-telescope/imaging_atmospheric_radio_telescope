import tarfile
import io


class TarStream:
    """
    TarStream is a simplified and more restricted writer/reader for tar-files.
    It always writes/reads files one-by-one in/from a stream without seeking.
    It only uses the filenam to identify a file.
    Permissions and dates are neglected.
    """

    def __init__(self, path, mode="r|"):
        self.path = str(path)
        if mode in ["r", "rb", "rb|"]:
            self.mode = "r|"
        elif mode in ["w", "wb", "wb|"]:
            self.mode = "w|"
        else:
            raise AssertionError
        self.tar = tarfile.open(name=self.path, mode=self.mode)

    def write(self, filename, filebytes):
        """
        Adds a file to tar-file with name 'filename' and payload 'filebytes'.
        """
        append_file(tar=self.tar, filename=filename, filebytes=filebytes)

    def read(self):
        """
        Reads the next file from the tar-file and returns its
        'filename' and 'filebytes' in a 2-tuple.
        """
        fileinfo = self.tar.next()
        if fileinfo is None:
            raise StopIteration
        filebytes = self.tar.extractfile(fileinfo).read()
        return fileinfo.name, filebytes

    def close(self):
        self.tar.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        out = "{:s}(path='{:s}')".format(self.__class__.__name__, self.path)
        return out


def append_file(tar, filename, filebytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(filename)
        info.size = buff.write(filebytes)
        buff.seek(0)
        tar.addfile(info, buff)
