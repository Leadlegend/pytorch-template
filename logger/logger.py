import os
import json
import logging
import logging.config

from pathlib import Path
from collections import OrderedDict


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def setup_logging(save_dir='.', log_config='code/logger/config.json', default_level=logging.INFO, file_name=None):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                if file_name is not None:
                    handler['filename'] = os.path.join(
                        save_dir, file_name)
                else:
                    handler['filename'] = os.path.join(
                        save_dir, handler['filename'])

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
