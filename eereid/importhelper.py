


class importhelper():

    def __init__(self, lib, job, install=None):
        self.lib = lib
        self.job = job
        if install is None: install = f"pip install {lib}"
        self.install = install

    def __getattr__(self, name):
        raise ImportError(f"""Using the '{self.job}' module requires the '{self.lib}' library. Please install it by running: '{self.install}'""")

