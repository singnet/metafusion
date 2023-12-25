import time

from .worker_base import ServiceThreadBase
from .prompting import Cfgen
from .sessions import GenSession
from .pipes import Prompt2ImPipe


class ServiceThread(ServiceThreadBase):
    def get_pipeline(self, pipe_name, model_id):
        pipe_class = self.get_pipe_class(pipe_name)
        pipeline = self._loader.load_pipeline(pipe_class._class, model_id)

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
                with self._lock:
                    data = self.queue[-1]
                    sess = self.sessions[data["session_id"]]
                self.logger.info("GENERATING: " + str(data))
                if 'start_callback' in data:
                    data['start_callback']()

                pipe_name = sess.get('pipe', 'Prompt2ImPipe')
                model_id = str(self.cwd/self.config["model_dir"]/self.models['base'][sess["model"]])
                pipe = self.get_pipeline(pipe_name, model_id)

                # TODO: the list of provided images can depend on the pipeline (e.g., it will be different for Cond)
                images = data['images']
                if images:
                    pipe.setup(**data, original_image=str(images[0]),
                            image_painted=str(images[1]))
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
                with self._lock:
                    self.queue.pop()
            time.sleep(0.2)
