import loguru
import os

class MyLogger:
    def __init__(self, output_dir=None):
        self.logger = loguru.logger
        if output_dir:
            self.logger.add(os.path.join(output_dir, 'out.log'))

        self.my_rank = int(os.environ.get("RANK", "0"))

    def info(self, message, main_process_only=True):
        if main_process_only:
            if self.my_rank == 0:
                self.logger.info(message)
        else:
            self.logger.info(message)

    def info_dic_step(self, dict, step, main_process_only=True):
        _str = "step: " + str(step) + "\t"
        for k, v in dict.items():
            if isinstance(v, float):
                _str += f"{k}: {v:.4f}\t"
            else:
                _str += f"{k}: {v}\t"
        self.info(_str, main_process_only)


    def warning(self, message, main_process_only=True):
        if main_process_only:
            if self.my_rank == 0:
                self.logger.warning(message)
        else:
            self.logger.warning(message)

    def error(self, message, main_process_only=True):
        if main_process_only:
            if self.my_rank == 0:
                self.logger.error(message)
        else:
            self.logger.error(message)

    def success(self, message, main_process_only=True):
        if main_process_only:
            if self.my_rank == 0:
                self.logger.success(message)
        else:
            self.logger.success(message)

if __name__ == "__main__":
    logger = MyLogger()
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.success("This is a success message.")