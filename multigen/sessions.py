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
        # Check if sequential CPU offloading is enabled
        self.offload_gpu_id = getattr(pipe, 'offload_gpu_id', None)

    def get_last_conf(self):
        conf = {**self.last_conf}
        conf.update(self.pipe.get_config())
        conf.update({
            'feedback': '?',
            'cversion': "0.0.1"})
        return conf
    
    def get_file_prefix(self, index):
        idxs = self.name_prefix + str(index).zfill(5)
        f_prefix = os.path.join(self.session_dir, idxs)
        if os.path.isfile(f_prefix + ".txt"):
            cnt = 1
            while os.path.isfile(f_prefix + "_" + str(cnt) + ".txt"):
                cnt += 1
            f_prefix += "_" + str(cnt)
        return f_prefix

    def save_conf(self, index, conf):
        cfg_name = self.get_file_prefix(index) + ".txt"
        with open(cfg_name, 'w') as f:
            print(json.dumps(conf, indent=4), file=f)
    
    def gen_sess(self, add_count=0, save_img=True, drop_cfg=False, 
                 force_collect=False, callback=None, save_metadata=False):
        """
        Run image generation session.
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
        images = []

        if save_img:
            os.makedirs(self.session_dir, exist_ok=True)

        # Determine batch size
        if self.offload_gpu_id is not None:
            # Sequential CPU offloading is enabled, set batch_size to a reasonable number
            batch_size = 8  # You can adjust this value based on your environment
        else:
            batch_size = 1  # Process one input at a time

        logging.info(f"Starting generation with batch_size = {batch_size}")
        confg_iter = iter(self.confg)
        index = self.confg.start_count

        while True:
            batch_inputs_list = []
            # Collect inputs into batch
            for _ in range(batch_size):
                try:
                    inputs = next(confg_iter)
                except StopIteration:
                    break  # No more inputs
                batch_inputs_list.append(inputs)

            if not batch_inputs_list:
                break  # All inputs have been processed

            # Prepare batch inputs
            batch_inputs_dict = {}
            for key in batch_inputs_list[0]:
                batch_inputs_dict[key] = [input[key] for input in batch_inputs_list]

            # Adjust 'generator' field with manual seeds
            batch_generators = []
            for seed in batch_inputs_dict.get('generator', [None] * len(batch_inputs_list)):
                if seed is not None:
                    batch_generators.append(torch.Generator().manual_seed(seed))
                else:
                    batch_generators.append(torch.Generator())
            batch_inputs_dict['generator'] = batch_generators

            # Generate images
            batch_images = self.pipe.gen(batch_inputs_dict)

            # Process generated images
            for i, image in enumerate(batch_images):
                idx = index + i
                self.last_index = idx
                self.last_conf = {**batch_inputs_list[i % len(batch_inputs_list)]}
                self.last_conf.update(self.pipe.get_config())
                self.last_conf.update({'feedback': '?', 'cversion': '0.0.1'})

                if save_img:
                    f_prefix = self.get_file_prefix(idx)
                    img_name = f_prefix + ".png"
                    exif = None
                    if save_metadata:
                        exif = util.create_exif_metadata(image, json.dumps(self.get_last_conf()))
                    image.save(img_name, exif=exif)
                    self.last_img_name = img_name
                    if not drop_cfg:
                        # Save configuration
                        self.save_conf(idx, self.get_last_conf())
                if not save_img or force_collect:
                    images.append(image)
                if callback is not None:
                    logging.debug("Call callback after generation")
                    callback()

            index += len(batch_images)
            logging.debug(f"Processed batch up to index {index}")

        logging.debug(f"Generation session completed.")
        return images if images else None
