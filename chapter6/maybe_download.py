# -*- coding: utf-8 -*-

import os
import zipfile
import random
import urllib.request


def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url+filename,filename)

    statinfo = os.stat(filename)
    
    if statinfo.st_size == expected_bytes:
        print("Found and Verified",filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            "Failed to verify " + filename + ". Can you get to it with a browser?")

    return filename

if __name__ == "__main__":
        
    url = "http://mattmahoney.net/dc/"
    filename = maybe_download("text8.zip",31344016)

    print("Download Finshed!")

