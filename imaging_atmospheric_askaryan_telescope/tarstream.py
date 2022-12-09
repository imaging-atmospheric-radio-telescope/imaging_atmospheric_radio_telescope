import tarfile
import io


class TarStream:
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
        append_file(tar=self.tar, filename=filename, filebytes=filebytes)

    def read(self):
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
