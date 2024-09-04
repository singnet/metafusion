import torch
import random
import time
import concurrent
from queue import Empty

from .worker_base import ServiceThreadBase
from .prompting import Cfgen
from .sessions import GenSession
from .pipes import Prompt2ImPipe, ModelType 

SDXL = 'SDXL'
FLUX = 'FLUX'
SD = 'SD'


class ServiceThread(ServiceThreadBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gpu_jobs = dict()

    def get_pipeline(self, pipe_name, model_id, model_type, cnet=None):
        pipe_class = self.get_pipe_class(pipe_name)
        return self._get_pipeline(pipe_class, model_id, model_type, cnet=cnet)

    def _get_device(self, model_id):
        # choose random from resting gpus
        # if there is no resting gpus choose
        # one with our model_id otherwise choose random
        devices = list(range(torch.cuda.device_count()))
        self.logger.debug('awailable devices %s', devices)
        if not devices:
            self.logger.debug('returning cpu device')
            return torch.device('cpu')
        with self._lock:
            self.logger.debug('locked gpu %s', self._gpu_jobs)
            free_gpus = [x for x in devices if x not in self._gpu_jobs]
            self.logger.debug('free gpus %s', free_gpus)
            if free_gpus:
                idx = random.choice(free_gpus)
            else:
                self.logger.debug('no free gpus')
                gpus_with_model = self._loader.get_gpu(model_id)
                if gpus_with_model:
                    idx = random.choice(gpus_with_model)
                else:
                    idx = random.choice(devices)
            self._gpu_jobs[idx] = model_id
            self.logger.debug(f'locked device cuda:{idx} for {model_id}')
            return torch.device('cuda', idx)

    def _get_pipeline(self, pipe_class, model_id, model_type, cnet=None):
        device = self._get_device(model_id)
        offload_device = None
        if cnet is None:
            # no controlnet
            if model_type == ModelType.SDXL:
                cls = pipe_class._classxl
            elif model_type == ModelType.FLUX:
                cls = pipe_class._flux
                if device.type == 'cuda':
                    offload_device = device.index
                    device = torch.device('cpu')
            else:
                cls = pipe_class._class
            pipeline = self._loader.load_pipeline(cls, model_id, torch_dtype=torch.bfloat16, 
                                                  device=device)
            self.logger.debug(f'requested {cls} {model_id} on device {device}, got {pipeline.device}')
            assert pipeline.device == device
            pipe = pipe_class(model_id, pipe=pipeline, device=device, offload_device=offload_device)
            if offload_device is None:
                assert pipeline.device == device
        else:
            # our pipeline uses controlnet
            pipeline = self._loader.get_pipeline(model_id, device=device, model_type=model_type)
            if model_type == FLUX:
                cnet_type = ModelType.FLUX
                if device.type == 'cuda':
                    offload_device = device.index
                    device = torch.device('cpu')
            if pipeline is None or 'controlnet' not in pipeline.components:
                # reload
                pipe = pipe_class(model_id, ctypes=[cnet], model_type=model_type, device=device, offload_device=offload_device)
                self._loader.cache_pipeline(pipe.pipe, model_id)
            else:
                pipe = pipe_class(model_id, pipe=pipeline, model_type=model_type, device=device, offload_device=offload_device)
        return pipe

    def run(self):
        self.logger.debug('running thread')
        num_of_workers = torch.cuda.device_count()
        if num_of_workers == 0:
            num_of_workers = 1
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_of_workers)
        while not self._stop:
            try:
                data = self.queue.get(block=False)
                self.logger.debug('submitting job %s', data)
                executor.submit(self.worker, data)
            except Empty:
                time.sleep(0.2)

    def worker(self, data):
        def _update(sess, job, gs):
            sess["images"].append(gs.last_img_name)
            if 'image_callback' in data:
                data['image_callback'](gs.last_img_name)
            job["count"] -= 1
        device = None
        # keep the job in the queue until complete
        try:
            session_id = data["session_id"]
            sess = self.sessions[session_id]
            sess['status'] ='running'
            self.logger.info("GENERATING: " + str(data))
            if 'start_callback' in data:
                data['start_callback']()

            pipe_name = sess.get('pipe', 'Prompt2ImPipe')
            model_id = str(self.cwd/self.config["model_dir"]/self.models['base'][sess["model"]]['id'])
            loras = [str(self.cwd/self.config["model_dir"]/'Lora'/self.models['lora'][x]['id']) for x in sess.get("loras", [])]
            data['loras'] = loras
            mt = self.models['base'][sess["model"]]['type']
            if mt == SDXL:
                model_type = ModelType.SDXL
            elif mt == SD:
                model_type = ModelType.SD
            elif mt == FLUX:
                model_type = ModelType.FLUX
            else:
                raise RuntimeError(f"unexpected model type {mt}")
            pipe = self.get_pipeline(pipe_name, model_id, model_type, cnet=data.get('cnet', None))
            device = pipe.pipe.device
            offload_device = pipe.offload_gpu_id
            self.logger.debug(f'running job on {device} offload {offload_device}')
            if device.type in  ['cuda', 'meta']:
                with self._lock:
                    if device.type == 'meta':
                        self._gpu_jobs[offload_device] = model_id
                    else:
                        self._gpu_jobs[device.index] = model_id
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
            nprompt_default = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, nude, bad hands, duplicate heads, bad anatomy, bad crop"
            nprompt = data.get('nprompt', nprompt_default)
            seeds = data.get('seeds', None)
            self.logger.debug(f"offload_device {pipe.offload_gpu_id}")
            gs = GenSession(self.get_image_pathname(data["session_id"], None),
                            pipe, Cfgen(data["prompt"], nprompt, seeds=seeds))
            gs.gen_sess(add_count = data["count"],
                        callback = lambda: _update(sess, data, gs))
            if 'finish_callback' in data:
                data['finish_callback']()
        except (RuntimeError, TypeError, NotImplementedError) as e:
            self.logger.error("error in generation", exc_info=e)
            self.logger.error(f"offload_device {pipe.pipe._offload_gpu_id}")
            if 'finish_callback' in data:
                data['finish_callback']("Can't generate image due to error")
        except Exception as e:
            self.logger.error("unexpected error in generation", exc_info=e)
            raise e
        finally:
            with self._lock:
                index = None
                if device is not None and device.type == 'cuda':
                    index = device.index
                if pipe.pipe._offload_gpu_id is not None:
                    index = pipe.pipe._offload_gpu_id
                if index is not None:
                    self.logger.debug('unlock device %s', index)
                    del self._gpu_jobs[index]
