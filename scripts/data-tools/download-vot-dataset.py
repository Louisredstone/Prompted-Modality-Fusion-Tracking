# This script is copied from: https://blog.csdn.net/laizi_laizi/article/details/122492396
# (and modified)

import os
from tqdm import tqdm
import six
import csv
import requests
from urllib.parse import urlparse, urljoin
import tempfile
import shutil

# see: https://github.com/votchallenge/toolkit/tree/master/vot/stack

VOT_URLs = {
    "2013": "http://data.votchallenge.net/vot2013/dataset/description.json",
    "2014": "http://data.votchallenge.net/vot2014/dataset/description.json",
    "2015": "http://data.votchallenge.net/vot2015/dataset/description.json",
    "2015-TIR": "http://www.cvl.isy.liu.se/research/datasets/ltir/version1.0/ltir_v1_0_8bit.zip",
    "2016": "http://data.votchallenge.net/vot2016/main/description.json",
    "2016-TIR": "http://data.votchallenge.net/vot2016/vot-tir2016.zip",
    "2017": "http://data.votchallenge.net/vot2017/main/description.json",
    "2018-ST": "http://data.votchallenge.net/vot2018/main/description.json",
    "2018-LT": "http://data.votchallenge.net/vot2018/longterm/description.json",
    "2019-ST": "http://data.votchallenge.net/vot2019/main/description.json",
    "2019-LT": "http://data.votchallenge.net/vot2019/longterm/description.json",
    "2019-RGBD": "http://data.votchallenge.net/vot2019/rgbd/description.json",
    "2019-RGBT": "http://data.votchallenge.net/vot2019/rgbtir/meta/description.json",
    "2020-ST": "https://data.votchallenge.net/vot2020/shortterm/description.json",
    "2020-RGBT": "http://data.votchallenge.net/vot2020/rgbtir/meta/description.json",
    "2021-ST": "https://data.votchallenge.net/vot2021/shortterm/description.json",
    "2022-RGBD": "https://data.votchallenge.net/vot2022/rgbd/description.json",
    "2022-Depth": "https://data.votchallenge.net/vot2022/depth/description.json",
    "2022-STB": "https://data.votchallenge.net/vot2022/stb/description.json",
    "2022-STS": "https://data.votchallenge.net/vot2022/sts/description.json",
    "2022-LT": "https://data.votchallenge.net/vot2022/lt/description.json",
    "test": "http://data.votchallenge.net/toolkit/test.zip",
    "segmentation": "http://box.vicos.si/tracking/vot20_test_dataset.zip"
}

def download_dataset(name, path=".", existed=None):
    # name: eg. 2020-ST
    # path: eg. xxx/workspace/sequences
    if not name in VOT_URLs:
        raise ValueError(f"Unknown dataset '{name}'")
    url = VOT_URLs[name]
    print(f"Downloading dataset [{name}] to {path} with url='{url}'")
    if not url.endswith('.json'):
        download_uncompress(url, path)
        return
    # url: "https://data.votchallenge.net/vot2020/shortterm/description.json"
    meta = download_json(url)

    print(f'Downloading sequence dataset "{meta["name"]}" with {len(meta["sequences"])} sequences.')

    base_url = get_base_url(url) + "/"  # "http://data.votchallenge.net/vot2019/rgbd/"

    with Progress("Downloading", len(meta["sequences"])) as progress:
        for sequence in meta["sequences"]:
            if existed is not None and sequence["name"] in existed:
                continue
            sequence_directory = os.path.join(path, sequence["name"])
            os.makedirs(sequence_directory, exist_ok=True)

            data = {'name': sequence["name"], 'fps': sequence["fps"], 'format': 'default'}

            annotations_url = join_url(base_url, sequence["annotations"][
                "url"])  # 'https://zenodo.org/record/2640900/files/backpack_blue.zip'

            try:
                download_uncompress(annotations_url,
                                    sequence_directory)  # xxx/workspace/sequences/backpack_robotarm_lab_occ
            except Exception as e:
                raise Exception("Unable do download annotations bundle")
            except IOError as e:
                raise Exception(
                    "Unable to extract annotations bundle, is the target directory writable and do you have enough space?")

            for cname, channel in sequence["channels"].items():
                channel_directory = os.path.join(sequence_directory, cname)
                os.makedirs(channel_directory, exist_ok=True)

                channel_url = join_url(base_url, channel["url"])

                try:
                    download_uncompress(channel_url, channel_directory)
                except Exception as e:
                    raise Exception("Unable do download channel bundle")
                except IOError as e:
                    raise Exception(
                        "Unable to extract channel bundle, is the target directory writable and do you have enough space?")

                if "pattern" in channel:
                    data["channels." + cname] = cname + os.path.sep + channel["pattern"]
                else:
                    data["channels." + cname] = cname + os.path.sep

            write_properties(os.path.join(sequence_directory, 'sequence'), data)

            progress.relative(1)

    with open(os.path.join(path, "list.txt"), "w") as fp:
        for sequence in meta["sequences"]:
            fp.write('{}\n'.format(sequence["name"]))

def download_json(url):
    try:
        return requests.get(url).json()
    except requests.exceptions.RequestException as e:
        raise Exception("Unable to read JSON file {}".format(e))

def get_base_url(url):
    return url.rsplit('/', 1)[0]

def is_absolute_url(url):
    return bool(urlparse(url).netloc)

def join_url(url_base, url_path):
    if is_absolute_url(url_path):
        return url_path
    return urljoin(url_base, url_path)

def extract_files(archive, destination, callback = None):
    from zipfile import ZipFile

    with ZipFile(file=archive) as zip_file:
        # Loop over each file
        total=len(zip_file.namelist())
        for file in zip_file.namelist():

            # Extract each file to another directory
            # If you want to extract to current working directory, don't specify path
            zip_file.extract(member=file, path=destination)
            if callback:
                callback(1, total)

def download_uncompress(url, path):
    _, ext = os.path.splitext(urlparse(url).path)
    tmp_file = tempfile.mktemp(suffix=ext)
    try:
        download(url, tmp_file)
        extract_files(tmp_file, path)
    finally:
        if os.path.exists(tmp_file):
            os.unlink(tmp_file)


def download(url, output, callback=None, chunk_size=1024 * 32):
    with requests.session() as sess:

        while True:
            res = sess.get(url, stream=True)

            if not res.status_code == 200:
                raise Exception("File not available")

            if 'Content-Disposition' in res.headers:
                # This is the file
                break
            break

        if output is None:
            output = os.path.basename(url)

        output_is_path = isinstance(output, str)

        if output_is_path:
            tmp_file = tempfile.mktemp()
            filehandle = open(tmp_file, 'wb')
        else:
            tmp_file = None
            filehandle = output

        try:
            total = res.headers.get('Content-Length')

            if total is not None:
                total = int(total)

            for chunk in res.iter_content(chunk_size=chunk_size):
                filehandle.write(chunk)
                if callback:
                    callback(len(chunk), total)
            if tmp_file:
                filehandle.close()
                shutil.copy(tmp_file, output)
        except IOError:
            raise Exception("Error when downloading file")
        finally:
            try:
                if tmp_file:
                    os.remove(tmp_file)
            except OSError:
                pass

        return output

class Progress(object):
    class StreamProxy(object):

        def write(self, x):
            # Avoid print() second call (useless \n)
            if len(x.rstrip()) > 0:
                tqdm.write(x)

        def flush(self):
            # return getattr(self.file, "flush", lambda: None)()
            pass

    @staticmethod
    def logstream():
        return Progress.StreamProxy()

    def __init__(self, description="Processing", total=100):
        silent = False

        if not silent:
            self._tqdm = tqdm(disable=None,
                              bar_format=" {desc:20.20} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]")
            self._tqdm.desc = description
            self._tqdm.total = total
        if silent or self._tqdm.disable:
            self._tqdm = None
            self._value = 0
            self._total = total if not silent else 0

    def _percent(self, n):
        return int((n * 100) / self._total)

    def absolute(self, value):
        if self._tqdm is None:
            if self._total == 0:
                return
            prev = self._value
            self._value = max(0, min(value, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._tqdm.update(value - self._tqdm.n)  # will also set self.n = b * bsize

    def relative(self, n):
        if self._tqdm is None:
            if self._total == 0:
                return
            prev = self._value
            self._value = max(0, min(self._value + n, self._total))
            if self._percent(prev) != self._percent(self._value):
                print("%d %%" % self._percent(self._value))
        else:
            self._tqdm.update(n)  # will also set self.n = b * bsize

    def total(self, t):
        if self._tqdm is None:
            if self._total == 0:
                return
            self._total = t
        else:
            if self._tqdm.total == t:
                return
            self._tqdm.total = t
            self._tqdm.refresh()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._tqdm:
            self._tqdm.close()


def write_properties(filename, dictionary, delimiter='='):
    ''' Writes the provided dictionary in key sorted order to a properties
        file with each line in the format: key<delimiter>value
            filename -- the name of the file to be written
            dictionary -- a dictionary containing the key/value pairs.
    '''
    open_kwargs = {'mode': 'w', 'newline': ''} if six.PY3 else {'mode': 'wb'}
    with open(filename, **open_kwargs) as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter, escapechar='\\',
                            quoting=csv.QUOTE_NONE)
        writer.writerows(sorted(dictionary.items()))

import argparse

parser = argparse.ArgumentParser(description='Download VOT dataset. All available datasets: 2013, 2014, 2015, 2015-TIR, 2016, 2016-TIR, 2017, 2018-ST, 2018-LT, 2019-ST, 2019-LT, 2019-RGBD, 2019-RGBT, 2020-ST, 2020-RGBT, 2021-ST, 2022-RGBD, 2022-Depth, 2022-STB, 2022-STS, 2022-LT, test, segmentation')
parser.add_argument('name', type=str, help='Name of the dataset to download. (use "," to separate multiple datasets, use "all" to download all datasets)')
parser.add_argument('--vot-root', type=str, default='datasets/VOT', help='Path to save the dataset.')

if __name__ == '__main__':
    args = parser.parse_args()
    name = args.name
    vot_root = args.vot_root
    # Existed_Seqs = ['agility', 'ants1', 'ball2', 'ball3', 'basketball', 'birds1', 'bolt1', 'book', 'butterfly', 'car1']
    # download_dataset(name=NAME, path=SAVE_DIR, existed=Existed_Seqs)
    if name == 'all':
        names = VOT_URLs.keys()
    else:
        names = name.split(',')
    for name in names:
        if name not in VOT_URLs:
            print(f'URL of dataset {name} not found.')
            continue
        path = os.path.join(vot_root, name)
        if os.path.exists(path) and os.listdir(path):
            print(f'Path {path} already exists, skip downloading.')
            print('If you need to re-download, consider manually delete the folder first.')
        download_dataset(name=name, path=path)
