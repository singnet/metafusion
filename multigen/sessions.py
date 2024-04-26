import torch
import os
import json
from . import util
from .prompting import Cfgen


class GenSession:

    def __init__(self, session_dir, pipe, config: Cfgen, name_prefix=""):
        self.session_dir = session_dir
        self.pipe = pipe
        self.model_id = pipe.model_id
        self.confg = config
        self.last_conf = None
        self.name_prefix = name_prefix

    def get_last_conf(self):
        conf = {**self.last_conf}
        conf.update(self.pipe.get_config())
        conf.update({
            'feedback': '?',
            'cversion': "0.0.1"})
        return conf

    def get_last_file_prefix(self):
        idxs = self.name_prefix + str(self.last_index).zfill(5)
        f_prefix = os.path.join(self.session_dir, idxs)
        if os.path.isfile(f_prefix + ".txt"):
            cnt = 1
            while os.path.isfile(f_prefix + "_" + str(cnt) + ".txt"):
                cnt += 1
            f_prefix += "_" + str(cnt)
        return f_prefix

    def save_last_conf(self):
        self.last_cfg_name = self.get_last_file_prefix() + ".txt"
        with open(self.last_cfg_name, 'w') as f:
            print(json.dumps(self.get_last_conf(), indent=4), file=f)

    def gen_sess(self, add_count = 0, save_img=True,
                 drop_cfg=False, force_collect=False,
                 callback=None, save_metadata=False):
        self.confg.max_count += add_count
        self.confg.start_count = self.confg.count
        self.last_img_name = None
        self.last_cfg_name = None
        images = None
        if save_img:
            os.makedirs(self.session_dir, exist_ok=True)
        # collecting images to return if requested or images are not saved
        if not save_img or force_collect:
            images = []
        for inputs in self.confg:
            self.last_index = self.confg.count - 1
            self.last_conf = {**inputs}
            # TODO: multiple inputs?
            inputs['generator'] = torch.Generator().manual_seed(inputs['generator'])

            image = self.pipe.gen(inputs)
            if save_img:
                self.last_img_name = self.get_last_file_prefix() + ".png"
                exif = None
                if save_metadata:
                    exif = util.create_exif_metadata(image, json.dumps(self.get_last_conf()))
                image.save(self.last_img_name, exif=exif)
            if not save_img or force_collect:
                images += [image]
            # saving cfg only if images are saved and dropping is not requested
            if save_img and not drop_cfg:
                self.save_last_conf()
            if callback is not None:
                callback()
        return images
