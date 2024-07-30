import sys
import debias
import logging
from datetime import datetime

class MyStreamHandler(logging.FileHandler):
    def emit(self, record):
        record.msg = f'{datetime.now()}: {record.msg}'
        super().emit(record)

class LoggerWriter:
    def __init__(self, level):
        self.level = level
        self.terminal = sys.stdout if level == logging.INFO else sys.stderr

    def write(self, message):
        if message.strip():
            self.level(message)

    def flush(self):
        pass

log_filename = "gam.log"
logging.basicConfig(level=logging.INFO, handlers=[MyStreamHandler(log_filename)])
logger = logging.getLogger()

sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)

def main(argv):
    models = ["lcm", "sdv1", "pixart_sigma"]
    model = models[0]
    server_address = "127.0.0.1:8197"
    print("Start Debias!")
    epochs = 3
    for epoch in range(1, epochs + 1):
        print("Epoch: ", epoch)
        # generator
        json_template_path = f"./workflow/{model}.json" # make sure you provide correct form of json
        if epoch == 1:
            prompt_file_path = "./debias/data/prompt_test.txt"
        else:
            prompt_file_path = "./GAM_result/prompt/" + f"prompt_epoch{epoch - 1}.txt"
        debias.generate_images(prompt_file_path,
                                json_template_path,
                                server_address)
        image_directory = "/data/hanjun/comfy2/output"
        output_directory = "./GAM_result/image/" + f"{model}/epoch{epoch}"
        debias.build_directory(prompt_file_path,
                            image_directory,
                            output_directory)
        # align
        main_directory = "./GAM_result/image/" + f"{model}/epoch{epoch}"
        output_directory = f"./GAM_result/record/align/{model}"
        debias.process_all_subdirs(main_directory,
                                output_directory, epoch)
        # discriminator
        ground_truth_path = "./debias/data/truth.json"
        generative_path = "./GAM_result/record/align/" + f"{model}/epoch{epoch}.json"
        origin_txt_path = "./debias/data/prompt_test.txt"
        origin_array_path = "./debias/data/origin_array.json"
        json_array_path = f"./GAM_result/record/discriminator/{model}"
        output_prompt_file_path= f"./GAM_result/prompt/"
        debias.discriminator(ground_truth_path,
                            generative_path,
                            origin_txt_path,
                            origin_array_path,
                            output_prompt_file_path,
                            json_array_path,
                            epoch)
    print("finished")

if __name__ == "__main__":
    main(sys.argv)