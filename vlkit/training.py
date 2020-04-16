import os, re, logging
from os.path import join, isdir

def get_logger(path="log.txt"):
    logger = logging.getLogger("Logger")
    file_handler = logging.FileHandler(path, "w")
    stdout_handler = logging.StreamHandler()
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    logformatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    stdout_handler.setFormatter(logformatter)
    file_handler.setFormatter(logformatter)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger

class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger("Logger")
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        logformatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')
        self.stdout_handler.setFormatter(logformatter)
        self.file_handler.setFormatter(logformatter)
        self.logger.setLevel(logging.INFO)
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()

def run_path(s):
    if s.endswith("/"):
        s = s[:-1]

    l = re.findall(r"^.*-run-([0-9]+)", s)

    if len(l) == 0:
        # s = "a/b"
        s += "-run-1"
        # s = "a/b-run-1"
        run_id = 1
    else:
        run_id = int(l[0])

    # run_id = 1
    while isdir(s):
        s = s.replace("-run-%d"%run_id, "-run-%d"%(run_id+1))
        run_id += 1

    return s

if __name__ == "__main__":

    s = "c-run-1"

    print(run_path(s))
