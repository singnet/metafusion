

class LoaderTestCase(TestCase):

    def test_loader(self):
        """
        Test that weights are shared for different pipelines loaded from the same
        checkpoint
        """
        loader = Loader()
        model_id = self.get_model()
        model_type = self.model_type()
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
        if 'device' not in self.device_args:
            self.device_args['device'] = device
        classes = self.get_cls_by_type(MaskedIm2ImPipe)
        # load inpainting pipe
        cls = classes[model_type]
        pipeline = loader.load_pipeline(cls, model_id, **self.device_args)
        inpaint = MaskedIm2ImPipe(model_id, pipe=pipeline,  **self.device_args)

        prompt_classes = self.get_cls_by_type(Prompt2ImPipe)
        # create prompt2im pipe
        cls = prompt_classes[model_type]
        device_args = dict(**self.device_args)
        device = device_args.get('device', None)
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda', 0)
            else:
                device = torch.device('cpu', 0)
            device_args['device'] = device
        
        pipeline = loader.load_pipeline(cls, model_id, **device_args)
        prompt2image = Prompt2ImPipe(model_id, pipe=pipeline, **device_args)
        prompt2image.setup(width=512, height=512, scheduler=self.schedulers[0], clip_skip=2, steps=5)
        if device.type == 'cuda':
            self.assertEqual(inpaint.pipe.unet.conv_out.weight.data_ptr(),
                         prompt2image.pipe.unet.conv_out.weight.data_ptr(),
                         "unets are different")
