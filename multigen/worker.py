import time

from .worker_base import ServiceThreadBase
from .prompting import Cfgen
from .sessions import GenSession
from .pipes import Prompt2ImPipe

class ServiceThread(ServiceThreadBase):
    def run(self):
        def _update(sess, job, gs):
            sess["images"].append(gs.last_img_name)
            job["count"] -= 1
        while not self._stop:
            while self.queue:
                # keep the job in the queue until complete
                with self._lock:
                    data = self.queue[-1]
                    sess = self.sessions[data["session_id"]]
                self.logger.info("GENERATING: " + str(data))
                # TODO: it might be possible to avoid model loading each time
                pipe = Prompt2ImPipe(str(self.cwd/self.config["model_dir"]/self.models['base'][sess["model"]]),
                                     lpw=sess["lpw"])
                pipe.setup()#TODO: width=768, height=768)
                # TODO: add negative prompt to parameters
                nprompt = "jpeg artifacts, blur, distortion, watermark, signature, extra fingers, fewer fingers, lowres, nude, bad hands, duplicate heads, bad anatomy, bad crop"
                gs = GenSession(self.get_image_pathname(data["session_id"], None),
                                pipe, Cfgen(data["prompt"], nprompt))
                gs.gen_sess(add_count = data["count"],
                            callback = lambda: _update(sess, data, gs))
                with self._lock:
                    self.queue.pop()
            time.sleep(0.2)
