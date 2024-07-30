from align import process_all_subdirs
import torchvision
import transformers
print(torchvision.__version__)
print(transformers.__version__)
main_directory = "/rsch/hanjun/VersusDebias/GAM_result/image/lcm/epoch1"
output_directory = "/rsch/hanjun/VersusDebias/output"
process_all_subdirs(main_directory, output_directory)