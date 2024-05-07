import time
import torch

from .worker_base import ServiceThreadBase
from .prompting import Cfgen
from .sessions import GenSession
from .pipes import Prompt2ImPipe


class ServiceThread(ServiceThreadBase):
    def get_pipeline(self, pipe_name, model_id, cnet=None, xl=False):
        pipe_class = self.get_pipe_class(pipe_name)
        return self._get_pipeline(pipe_class, model_id, cnet=cnet, xl=xl)

    def _get_pipeline(self, pipe_class, model_id, cnet=None, xl=False):
        if cnet is None:
            if xl:
                cls = pipe_class._classxl
            else:
                cls = pipe_class._class
            pipeline = self._loader.load_pipeline(cls, model_id, torch_dtype=torch.float16)
            pipe = pipe_class(model_id, pipe=pipeline)
        else:
            pipeline = self._loader.get_pipeline(model_id)
            if pipeline is None or 'controlnet' not in pipeline.components:
                pipe = pipe_class(model_id, ctypes=[cnet])
                self._loader.register_pipeline(pipe.pipe, model_id)
            else:
                pipe = pipe_class(model_id, pipe=pipeline)
        return pipe

    def run(self):
        def _update(sess, job, gs):
            sess["images"].append(gs.last_img_name)
            if 'image_callback' in data:
                data['image_callback'](gs.last_img_name)
            job["count"] -= 1
        while not self._stop:
            while self.queue:
                # keep the job in the queue until complete
                try:
                    with self._lock:
                        data = self.queue[-1]
                        sess = self.sessions[data["session_id"]]
                    self.logger.info("GENERATING: " + str(data))
                    if 'start_callback' in data:
                        data['start_callback']()

                    pipe_name = sess.get('pipe', 'Prompt2ImPipe')
                    model_id = str(self.cwd/self.config["model_dir"]/self.models['base'][sess["model"]]['id'])
                    is_xl = self.models['base'][sess["model"]]['xl']
                    pipe = self.get_pipeline(pipe_name, model_id, cnet=data.get('cnet', None), xl=is_xl)
                    class_name = str(pipe.__class__)
                    self.logger.debug(f'got pipeline {class_name}')

                    images = data['images']
                    if 'MaskedIm2ImPipe' in class_name:
                        pipe.setup(**data, original_image=str(images[0]),
                                image_painted=str(images[1]))
                    elif any([x in class_name for x in ('Im2ImPipe', 'Cond2ImPipe')]):
                        pipe.setup(**data, fimage=str(images[0]))
                    else:
                        pipe.setup(**data)
                    # TODO: add negative prompt to parameters
                    nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, nude, bad hands, duplicate heads, bad anatomy, bad crop"
                    gs = GenSession(self.get_image_pathname(data["session_id"], None),
                                    pipe, Cfgen(data["prompt"], nprompt))
                    gs.gen_sess(add_count = data["count"],
                                callback = lambda: _update(sess, data, gs))
                    if 'finish_callback' in data:
                        data['finish_callback']()
                except (RuntimeError, TypeError) as e:
                    if 'finish_callback' in data:
                        self.logger.error("error in generation", exc_info=e)
                        data['finish_callback']("Can't generate image due to error")
                finally:
                    with self._lock:
                        self.queue.pop()
            time.sleep(0.2)
