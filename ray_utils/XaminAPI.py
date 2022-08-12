import ray

class XaminAPI():
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.resources = kwargs.pop('resources')
        print(f"Resources: {self.resources}, {self.kwargs}")
    
    def __enter__(self):

        @ray.remote(**self.resources)
        def __ray_fn(*args, **kwargs):
            return self.func(*args, **kwargs)
        
        print(f"Ray Status: {ray.is_initialized()}")
        if ray.is_initialized():
            obj_ref = __ray_fn.remote(*self.args, **self.kwargs)
            result = ray.get(obj_ref)
        else:
            # TODO: Add logging warning for running locally.
            return self.func(*self.args, **self.kwargs)
        return result

    def __exit__(self, type, value, traceback):
        pass
        