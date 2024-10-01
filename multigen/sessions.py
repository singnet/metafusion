import torch
import os
import json
from . import util
from .prompting import Cfgen
import logging


class GenSession:

    def __init__(self, session_dir, pipe, config: Cfgen, name_prefix=""):
        """
        Initialize a GenSession instance.

        Args:
            session_dir (str):
                The directory to store the session files.
            pipe (Pipe):
                The pipeline object for generating images.
            config (Cfgen):
                The configuration object for the generation process.
            name_prefix (str, *optional*):
                The prefix for the generated file names. Defaults to "".
        """
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
        """
        Run image generation session

        Args:
            add_count (int, *optional*):
                The number of additional iterations to add. Defaults to 0.
            save_img (bool, *optional*):
                Whether to save the generated images on local filesystem. Defaults to True.
            drop_cfg (bool, *optional*):
                If true don't save configuration file for each image. Defaults to False.
            force_collect (bool, *optional*):
                Force returning generated images even if save_img is true. Defaults to False.
            callback (callable, *optional*):
                A callback function to be called after each iteration. Defaults to None.
            save_metadata (bool, *optional*):
                Whether to save metadata in the image EXIF. Defaults to False.

        Returns:
            List[Image.Image]: The generated images if `save_img` is False or `force_collect` is True.
        """
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
        logging.info(f"add count = {add_count}")
        jk = 0
        for inputs in self.confg:
            self.last_index = self.confg.count - 1
            self.last_conf = {**inputs}
            # TODO: multiple inputs?
            inputs['generator'] = torch.Generator().manual_seed(inputs['generator'])
            logging.debug("start generation")
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
                logging.debug("call callback after generation")
                callback()
            jk += 1
        logging.debug(f"done iteration {jk}")
        return images
