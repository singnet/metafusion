import random

def _concat_kw(kw_desc, delim=' '):
    prompt = ''
    for kw in kw_desc:
        w = _get_one_kw_rec(kw)
        if w is None: continue
        if len(prompt) > 0:
            prompt += delim
        prompt += w
    return prompt

def _get_one_kw_rec(kw_desc):
    if isinstance(kw_desc, str):
        return kw_desc
    if isinstance(kw_desc, list):
        if kw_desc[0] == '+':
            return _concat_kw(kw_desc[1:])
        return _get_one_kw_rec(kw_desc[random.randint(0, len(kw_desc)-1)])
    if isinstance(kw_desc, dict):
        if random.random() < kw_desc['p']:
            return _get_one_kw_rec(kw_desc['w'])
        else:
            return None
    raise NotImplementedError

def get_prompt(prompt_desc):
    if isinstance(prompt_desc, str):
        return prompt_desc
    return _concat_kw(prompt_desc, delim=', ')


class Cfgen:

    def __init__(self, prompt, negative_prompt="", max_count=0, seeds=None, max_seed_rounds=1):
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.seeds = seeds
        # explicitly indicating the number of cycles over seeds
        self.max_seed_rounds = max_seed_rounds
        self.max_count = max_count
        self.count = 0
        self.start_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        nseeds = len(self.seeds) if self.seeds is not None else 0
        if (self.max_count > 0 and self.count == self.max_count) or \
           (nseeds > 0 and self.count - self.start_count == self.max_seed_rounds * nseeds) or \
           (self.max_count == 0 and nseeds == 0):
            raise StopIteration
        seed = self.seeds[self.count % nseeds] if nseeds > 0 else \
               random.randint(1, 1024*1024*1024*4-1)
        self.count += 1
        return {'prompt': get_prompt(self.prompt),
                'generator': seed,
                'negative_prompt': get_prompt(self.negative_prompt)}

